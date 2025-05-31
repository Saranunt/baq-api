# Weather Forecasting API
This repo is flying solo from the main one for one noble reason: we’re broke af. No AWS, no fancy cloud bills—just good ol' budget-friendly, lightweight deployment.

## Quick Start

```bash
# Run with Docker
docker build -t weather-api .

# Run with AWS credentials
docker run -d -p 9000:9000 \
  -e AWS_ACCESS_KEY_ID=your_access_key \
  -e AWS_SECRET_ACCESS_KEY=your_secret_key \
  weather-api

# Run without Docker
uvicorn app.main:app --host 0.0.0.0 --port 9000
```

## Docker Publishing

```bash
# Tag the image
docker tag weather-api saranunt/baq-api-final:latest

# Login to DockerHub
docker login

# Push to DockerHub
docker push saranunt/baq-api-final:latest

# Pull from DockerHub
docker pull saranunt/baq-api-final:latest
```

## Test API Endpoints

### Windows CMD Test Commands

```cmd
REM Test root endpoint
curl http://localhost:9000

REM Test prediction with 5-step forecast
curl -X POST http://localhost:9000/predict/next -H "Content-Type: application/json" -d "{\"forecast_horizon\": 5}"

REM Test prediction with default horizon (48 steps)
curl -X POST http://localhost:9000/predict/next -H "Content-Type: application/json" -d "{}"

REM Cache 96-step predictions to S3
curl -X POST http://localhost:9000/predict/cache
```

## API Documentation
- Swagger UI: http://localhost:9000/docs
- ReDoc: http://localhost:9000/redoc

## Endpoints
- `/predict/next`: Get predictions for specified horizon
- `/predict/cache`: Generate and cache 96-step predictions to S3 with timestamps 