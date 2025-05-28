FROM python:3.10-slim

# Set environment variables
ENV TF_ENABLE_ONEDNN_OPTS=0
ENV PYTHONUNBUFFERED=1
ENV AWS_ACCESS_KEY_ID=YOUR_KEY
ENV AWS_SECRET_ACCESS_KEY=YOUR_SECRET
ENV AWS_DEFAULT_REGION=ap-southeast-1

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy code and model files
COPY app/ /app/app/
COPY artifacts/ /app/artifacts/

# Expose FastAPI app port
EXPOSE 9000

# Run FastAPI using uvicorn
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "9000"]

