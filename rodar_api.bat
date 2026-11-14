@echo off
cd /d "%~dp0"
if not exist ".venv" python -m venv .venv
call .venv\Scripts\activate.bat
pip install -q -r requirements.txt
echo Abra http://127.0.0.1:8000  (front)  e  http://127.0.0.1:8000/docs  (Swagger)
python -m uvicorn api.main:app --reload
pause
