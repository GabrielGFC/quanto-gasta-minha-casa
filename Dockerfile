FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY ENB2012_data.xlsx .
COPY api ./api
COPY front ./front

# treina os modelos durante o build: o servidor sobe rápido e os .pkl ficam dentro da imagem
RUN python -m api.treinamento

RUN useradd --create-home app && chown -R app /app
USER app

EXPOSE 8000
CMD ["sh", "-c", "uvicorn api.main:app --host 0.0.0.0 --port ${PORT} --proxy-headers"]
