FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt gunicorn

# Copy app files
COPY . .

# Cloud Run sets PORT env var
ENV PORT=8080
ENV PYTHONUNBUFFERED=1

# Run with gunicorn (production WSGI server)
CMD exec gunicorn --bind :$PORT --workers 1 --threads 2 --timeout 120 app:app
