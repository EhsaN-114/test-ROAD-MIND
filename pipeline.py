"""The per-junction processing loop.

Every frame (in a background thread):
    read latest frame -> detect + track (every Nth frame) -> per-road stats
    -> people estimate -> confidence check -> draw overlays -> publish state

Once per advice cycle (or when the officer asks, or on emergency):
    advisor -> Bangla/English explanation -> simulated comparison -> forecast
    -> decision log entry (pending until the officer accepts or overrides)

Detection runs live on every run; nothing is pre-recorded or replayed.
"""
from __future__ import annotations

import os
import threading
import time
import traceback

import cv2
import supervision as sv

from app import db
from app.config import CYCLE_SECONDS, DETECT_EVERY, DISPLAY_MAX_FPS, JPEG_QUALITY, MODEL_PATH
from app.core.advisor import AdvisorSettings, advise
from app.core.explain import explain
from app.core.people import add_people, merged_occupancy, rank
from app.core.rhythm import day_kind, forecast, now_local
from app.core.simulator import compare
from app.video.annotate import annotate
from app.video.confidence import ConfidenceChecker
from app.video.detector import Detector
from app.video.source import VideoSource
from app.video.tracker import JunctionTracker

ADVICE_INTERVAL = int(os.getenv("ROADMIND_ADVICE_INTERVAL", "0"))  # 0 = once per signal cycle
EMERGENCY_SECONDS = 60
WARMUP_SECONDS = 4
COUNTS_EVERY_SECONDS = 60
FEATURES = ["people", "advice", "log", "confidence", "simulation", "forecast", "emergency"]


class JunctionPipeline:
    def __init__(self, junction: dict) -> None:
        self.junction = junction
        self.source = VideoSource(junction.get("source", ""))
        self.tracker = JunctionTracker(junction["roads"], junction["crossings"])
        self.confidence = ConfidenceChecker()
        self.occupancy = merged_occupancy(junction.get("occupancy"))
        self._lock = threading.Lock()
        self._jpeg: bytes | None = None
        self._state: dict = {"status": "starting"}
        self._advice_state: dict | None = None
        self._forecast: dict | None = None
        self._emergency: dict | None = None   # {"road": id, "until": t}
        self._refresh = threading.Event()
        self._running = False
        self._last_frame = None
        self._last_roads: list[dict] = []
        self._last_pedestrians = 0

    # --- lifecycle --------------------------------------------------------
    def start(self) -> "JunctionPipeline":
        self.source.start()
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()
        return self

    def stop(self) -> None:
        self._running = False
        self.source.stop()

    def reload(self, junction: dict) -> None:
        """New zones or occupancy values from the officer."""
        self.junction = junction
        self.tracker.set_zones(junction["roads"], junction["crossings"])
        self.occupancy = merged_occupancy(junction.get("occupancy"))
        self._refresh.set()

    # --- officer actions --------------------------------------------------
    def request_advice(self) -> None:
        self._refresh.set()

    def start_emergency(self, road_id: str) -> None:
        self._emergency = {"road": road_id, "until": time.time() + EMERGENCY_SECONDS}
        self._refresh.set()

    def cancel_emergency(self) -> None:
        self._emergency = None
        self._refresh.set()

    # --- outputs ----------------------------------------------------------
    def latest_jpeg(self) -> bytes | None:
        with self._lock:
            return self._jpeg

    def latest_state(self) -> dict:
        with self._lock:
            state = dict(self._state)
            state["advice"] = self._advice_state
            state["forecast"] = self._forecast
            if self._emergency:
                state["emergency"] = {"road": self._emergency["road"],
                                      "remaining_s": max(0, int(self._emergency["until"] - time.time()))}
            else:
                state["emergency"] = None
            return state

    # --- advice -----------------------------------------------------------
    def _settings(self) -> AdvisorSettings:
        return AdvisorSettings(cycle=int(self.junction.get("cycle_seconds", CYCLE_SECONDS)))

    def _make_advice(self, roads: list[dict], pedestrians: int, conf: dict, frame) -> None:
        settings = self._settings()
        emergency_road = self._emergency["road"] if self._emergency else None
        fixed_plan = self.junction.get("fixed_plan", {})
        advice = advise(roads, fixed_plan, pedestrians, conf["level"], conf["reasons"], emergency_road, settings).to_dict()
        text = explain(advice, use_llm=False)  # instant template; LLM version follows below

        simulation = None
        if advice["plan"]:
            sim_roads = []
            for r in roads:
                rate = r.get("arrivals_per_min", 0) / 60
                if rate <= 0:  # not enough tracking history yet: assume the queue refills once per cycle
                    rate = r.get("total_vehicles", 0) / max(settings.cycle, 1)
                sim_roads.append({"id": r["id"], "arrival_rate": rate,
                                  "ppv": r.get("people_per_vehicle") or self.occupancy["other"],
                                  "initial_waits": r.get("queued_waits", [])})
            fixed_greens = {r["id"]: int(fixed_plan.get(r["id"], settings.min_green)) for r in roads}
            plan_greens = {row["id"]: row["green"] for row in advice["plan"]}
            simulation = compare(sim_roads, fixed_greens, settings.base_walk,
                                 advice["order"], plan_greens, advice["walk_seconds"])

        thumb = db.save_thumbnail(frame, f"{self.junction['id']}_{int(time.time() * 1000)}") if frame is not None else None
        decision_id = db.add_decision(self.junction["id"], advice, text, thumb)
        with self._lock:
            self._advice_state = {"id": decision_id, **advice, "explanation": text, "simulation": simulation}

        def improve_text() -> None:
            better = explain(advice, use_llm=True)
            if better["source"] == "llm":
                db.update_explanation(decision_id, better)
                with self._lock:
                    if self._advice_state and self._advice_state.get("id") == decision_id:
                        self._advice_state = {**self._advice_state, "explanation": better}
        threading.Thread(target=improve_text, daemon=True).start()

    def _record_counts(self, roads: list[dict]) -> None:
        when = now_local()
        people = sum(r.get("people", 0) for r in roads)
        vehicles = sum(r.get("total_vehicles", 0) for r in roads)
        db.add_counts(self.junction["id"], when.isoformat(), day_kind(when), when.hour * 60 + when.minute,
                      people, vehicles, {r["id"]: r.get("people", 0) for r in roads})
        fc = forecast(db.counts_history(self.junction["id"]), people, when)
        with self._lock:
            self._forecast = fc

    # --- main loop --------------------------------------------------------
    def _loop(self) -> None:
        """Keep the video running: if one frame fails, report it and start again."""
        while self._running:
            try:
                self._run()
            except Exception as exc:  # noqa: BLE001 - never let the live view freeze silently
                traceback.print_exc()
                with self._lock:
                    self._state = {"status": "error", "error": f"Processing error: {type(exc).__name__}: {exc} (restarting)"}
                time.sleep(1)

    def _detect_loop(self, detector: Detector, alive: threading.Event, errors: list) -> None:
        """AI thread: detect + track the newest frame as fast as the CPU allows.

        It runs separately from the video drawing, so the dashboard video stays smooth
        even when YOLO is slow; boxes update at the AI speed ("AI fps").
        """
        last_id, done, times = -1, 0, []
        try:
            while alive.is_set() and self._running:
                frame, frame_id = self.source.read()
                if frame is None or frame_id == last_id or (DETECT_EVERY > 1 and frame_id - last_id < DETECT_EVERY):
                    time.sleep(0.005)
                    continue
                last_id = frame_id
                h, w = frame.shape[:2]
                t = time.perf_counter()
                detections = detector.detect(frame)
                tracked, stats = self.tracker.update(detections, w, h, t)
                confs = detections.confidence
                mean_conf = float(confs.mean()) if confs is not None and len(confs) else None
                stats["mean_confidence"] = round(mean_conf, 3) if mean_conf is not None else None
                stats["roads"] = add_people(stats["roads"], self.occupancy)
                done += 1
                conf = self.confidence.update(frame, len(detections), mean_conf) if done % 3 == 1 else None
                times.append(time.perf_counter())
                times = [x for x in times if times[-1] - x <= 3.0]
                ai_fps = (len(times) - 1) / max(times[-1] - times[0], 1e-6) if len(times) > 1 else 0.0
                with self._lock:
                    self._det = {"tracked": tracked, "stats": stats, "ai_fps": round(ai_fps, 1),
                                 "conf": conf or self._det.get("conf") or self.confidence.last}
        except Exception as exc:  # noqa: BLE001 - handed to the main loop, which restarts
            errors.append(exc)

    def _run(self) -> None:
        detector = Detector(MODEL_PATH)  # own detector so the tracker's memory starts fresh
        self._det = {"tracked": sv.Detections.empty(), "stats": {"roads": [], "crossings": []},
                     "ai_fps": 0.0, "conf": self.confidence.last}
        alive, errors = threading.Event(), []
        alive.set()
        threading.Thread(target=self._detect_loop, args=(detector, alive, errors), daemon=True).start()
        last_id, last_draw = -1, 0.0
        fps_window: list[float] = []
        started = None
        next_advice = None
        next_counts = time.time() + 30
        try:
            while self._running:
                if errors:
                    raise errors[0]
                frame, frame_id = self.source.read()
                if frame is None or frame_id == last_id or time.perf_counter() - last_draw < 1 / DISPLAY_MAX_FPS:
                    if frame is None and self.source.error:
                        with self._lock:
                            self._state = {"status": "error", "error": self.source.error}
                    time.sleep(0.005)
                    continue
                last_id, last_draw = frame_id, time.perf_counter()
                h, w = frame.shape[:2]
                with self._lock:
                    det = self._det
                tracked, stats, conf = det["tracked"], det["stats"], det["conf"]

                pedestrians = sum(c["people"] for c in stats.get("crossings", []))
                image = annotate(frame, tracked, self.tracker.roads, self.tracker.crossings, stats, self.occupancy)
                ok, buf = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])

                fps_window.append(time.perf_counter())
                fps_window = [x for x in fps_window if fps_window[-1] - x <= 2.0]
                fps = (len(fps_window) - 1) / max(fps_window[-1] - fps_window[0], 1e-6) if len(fps_window) > 1 else 0.0

                # --- advice timing: once per cycle, on request, or on emergency changes ---
                now = time.time()
                started = started or now
                if self._emergency and now > self._emergency["until"]:
                    self._emergency = None
                    self._refresh.set()
                interval = ADVICE_INTERVAL or int(self.junction.get("cycle_seconds", CYCLE_SECONDS))
                due = next_advice is None and now - started >= WARMUP_SECONDS and det["ai_fps"] > 0
                due = due or (next_advice is not None and now >= next_advice) or self._refresh.is_set()
                if due:
                    self._refresh.clear()
                    try:
                        self._make_advice(stats["roads"], pedestrians, conf, frame)
                    except Exception as exc:  # never let one bad advice stop the video
                        with self._lock:
                            self._advice_state = {"error": f"Advice failed: {type(exc).__name__}: {exc}"}
                    next_advice = now + interval
                if now >= next_counts:
                    next_counts = now + COUNTS_EVERY_SECONDS
                    threading.Thread(target=self._record_counts, args=(stats["roads"],), daemon=True).start()

                state = {
                    "status": "running",
                    "junction": {"id": self.junction["id"], "name": self.junction.get("name")},
                    "frame_size": [w, h],
                    "fps": round(fps, 1),
                    "ai_fps": det["ai_fps"],
                    "supported_types": detector.supported_types,
                    "occupancy": self.occupancy,
                    "features": FEATURES,
                    "confidence": conf,
                    "pedestrians_waiting": pedestrians,
                    "rank_people": rank(stats["roads"], "people"),
                    "rank_vehicles": rank(stats["roads"], "total_vehicles"),
                    "next_advice_in_s": max(0, int(next_advice - now)) if next_advice else None,
                    **stats,
                }
                with self._lock:
                    if ok:
                        self._jpeg = buf.tobytes()
                    self._state = state
        finally:
            alive.clear()


class PipelineManager:
    """Keeps one running pipeline (the selected junction) to save CPU."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._active: JunctionPipeline | None = None

    def get(self, junction: dict) -> JunctionPipeline:
        with self._lock:
            if self._active is None or self._active.junction["id"] != junction["id"]:
                if self._active is not None:
                    self._active.stop()
                self._active = JunctionPipeline(junction).start()
            return self._active

    def reload(self, junction: dict) -> None:
        with self._lock:
            if self._active is not None and self._active.junction["id"] == junction["id"]:
                if self._active.junction.get("source") != junction.get("source"):
                    # new camera or file: restart the video and the tracker
                    self._active.stop()
                    self._active = JunctionPipeline(junction).start()
                else:
                    self._active.reload(junction)

    def stop_all(self) -> None:
        with self._lock:
            if self._active is not None:
                self._active.stop()
                self._active = None
