FROM python:3.12-slim

WORKDIR /app

# Install packages before copying the code, so a code change does not reinstall them.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY gateway/ gateway/

CMD ["uvicorn", "gateway.main:app", "--host", "0.0.0.0", "--port", "8000"]
