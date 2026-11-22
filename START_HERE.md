# Roadmind — start here (local)

## First time (needs internet once)
```
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
copy .env.example .env
```
Optional: open `.env` and paste your Gemini key after `GEMINI_API_KEY=`.

## Every time
```
.venv\Scripts\activate
uvicorn app.main:app --reload
```
Open http://localhost:8000 — the **Test junction (stock clip)** opens by default, with its road zones already drawn.

## Switch videos
- Use the **Junction** dropdown at the top left.
- Your own video: open http://localhost:8000/calibrate → **Upload a video** → **Auto-detect roads** (or draw) → **Save**.

## Faster on a laptop (optional)
```
pip install openvino
python scripts/speed_up.py
```
Put the line it prints into `.env`, then restart.

## Important
`data/clips/junction_40s.mp4` is a watermarked stock preview from London — for practice only.
It is in `.gitignore`: never commit or publish it. For the final, use your own junction recording.
