#!/usr/bin/env bash
cd "$(dirname "$0")"
[ -d .venv ] || python3 -m venv .venv
source .venv/bin/activate
pip install -q -r requirements.txt
echo "Abra http://127.0.0.1:8000 (front) e http://127.0.0.1:8000/docs (Swagger)"
python -m uvicorn api.main:app --reload
