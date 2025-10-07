FROM python:3.11-slim

# pdf2image (Poppler)
RUN apt-get update && apt-get install -y --no-install-recommends \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .

RUN useradd -m appuser
USER appuser

ENV PORT=8080
EXPOSE 8080

CMD ["gunicorn", "app:app", "--bind=0.0.0.0:8080", "--workers=2", "--threads=4", "--timeout=180"]
