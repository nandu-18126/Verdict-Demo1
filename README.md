# VERDICT — Live Demo App (flat structure)

Upload/verdict/scoring app for the VERDICT pilot. This version has
**no subfolders** — every file sits directly in the repo root — because
browser-based drag-and-drop uploads (GitHub web UI, Hugging Face Spaces
web UI) frequently drop nested folder structure, which breaks Python's
package imports. Flat avoids that failure mode entirely.

## Files (all at root — upload every one of these directly, nothing nested)

```
Dockerfile
requirements.txt
main.py
model.py
scoring.py
index.html
README.md
```

## Uploading correctly

Whichever platform you use (GitHub, Render, Hugging Face Spaces), when you
drag files in: select all 6 files above and drop them in **one batch** at
the repo root. Do not create or upload a subfolder — there shouldn't be
one in this version at all.

## Deploying on Render

1. Push these files to a GitHub repo (root level, as above)
2. Render → New → Web Service → connect the repo
3. Render auto-detects the Dockerfile → Create Web Service
4. Build takes ~5-10 min (downloading torch + AASIST)

## Deploying on Hugging Face Spaces

Same files, but Spaces also needs this block at the very top of README.md
for it to recognize the Docker SDK:

```
---
title: Verdict Demo
emoji: 🎙️
colorFrom: gray
colorTo: yellow
sdk: docker
app_port: 7860
pinned: false
---
```

## Running locally with Docker

```bash
docker build -t verdict-app .
docker run -p 7860:7860 verdict-app
```
Open http://localhost:7860

## Calibration note

REAL_INDEX defaults to 1 (common AASIST/ASVspoof convention). Test with a
known-real clip first — if the verdict comes out backwards, fix it live:

```
POST /calibrate   (upload a known-real audio clip as real_file)
```

Or open `/docs` in your browser for a point-and-click test page (FastAPI's
built-in Swagger UI) — no coding needed.
