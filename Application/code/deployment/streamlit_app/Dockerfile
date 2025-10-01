FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt ./requirements.txt

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt streamlit requests

COPY Application ./Application

ENV PYTHONPATH=/app

EXPOSE 8501

# Allow overriding API host/port if needed
ENV API_HOST=api \
    API_PORT=8000

CMD ["streamlit", "run", "Application/code/deployment/streamlit_app/app.py", "--server.port=8501", "--server.address=0.0.0.0"]
