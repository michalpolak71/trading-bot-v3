FROM python:3.12.1-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY bot_ultimate.py .

CMD ["python", "-u", "bot_ultimate.py"]
