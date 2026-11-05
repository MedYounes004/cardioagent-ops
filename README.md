# CardioAgent Ops

[![MLOps CI/CD](https://github.com/MedYounes004/cardioagent-ops/actions/workflows/mlops-ci-cd.yml/badge.svg)](https://github.com/MedYounes004/cardioagent-ops/actions/workflows/mlops-ci-cd.yml)

CardioAgent Ops is a containerized pipeline for phonocardiogram (PCG) analysis. An uploaded WAV recording is validated by the API gateway, processed by the audio engine, and passed to a multi-agent orchestrator that produces a validated clinical analysis report.

> **Important:** This project is for research and engineering use. It is not a medical device and does not replace evaluation by a qualified clinician.

## Architecture

```text
Client
  |
  v
API Gateway (:8000)
  |-- validates WAV uploads and request size
  |-- serves the browser console at /
  |
  +--> Audio Engine (:8001)
  |      |-- extracts audio features
  |      +-- runs PCG inference
  |
  +--> Agent Orchestrator (:8002)
         |-- Analyst -> Cardiologist -> Auditor
         +-- optional LLM integration

MLflow Server (:5000) stores experiment metadata and artifacts.
```

## Quick start

### Prerequisites

- Docker with Docker Compose
- A WAV-format PCG recording for API testing
- Optional: an OpenAI-compatible or Ollama-compatible LLM endpoint for generated clinical assessment

Start the complete stack:

```bash
docker compose up --build
```

The gateway is available at <http://localhost:8000>. Open that URL for the browser console, or use the API directly. Stop the stack with:

```bash
docker compose down
```

MLflow is available at <http://localhost:5000>.

## API usage

Submit a PCG recording to the gateway:

```bash
curl -X POST http://localhost:8000/analyze_pcg \
  -F "file=@recording.wav;type=audio/wav" \
  -F "patient_id=patient-001" \
  -F 'metadata={"source":"clinic-a"}'
```

The response contains the request ID, model prediction and probabilities, risk level, analyst summary, clinical assessment, recommendations, audit result, and safety disclaimer. Only WAV files are accepted; uploads are limited to 25 MiB by default.

Readiness and liveness endpoints:

```bash
curl http://localhost:8000/health/live
curl http://localhost:8000/health/ready
```

The readiness endpoint checks the audio engine and agent orchestrator. Each internal service also exposes `GET /health`.

## Configuration

The following environment variables can be set in the shell or a `.env` file before starting Compose:

| Variable | Default | Description |
| --- | --- | --- |
| `LLM_URL` | empty | Optional LLM endpoint used by the agent orchestrator |
| `LLM_MODEL` | `cardio-qwen` | Model name sent to the LLM endpoint |
| `MAX_UPLOAD_BYTES` | `26214400` | Gateway upload limit |
| `DOWNSTREAM_TIMEOUT_SECONDS` | `60` | Gateway timeout for internal requests |
| `HEALTH_TIMEOUT_SECONDS` | `5` | Gateway dependency health-check timeout |
| `ALLOWED_ORIGINS` | `*` | Comma-separated CORS origins |
| `SERVICE_VERSION` | `1.0.0` | Version reported by the gateway and orchestrator |

The gateway's internal service URLs can also be overridden with `AUDIO_ENGINE_URL` and `ORCHESTRATOR_URL`.

## Development and testing

Run unit tests for an individual service from its directory after installing its requirements:

```bash
cd api-gateway
pip install -r requirements-dev.txt
pytest
```

Repeat for `audio-engine` and `agent-orchestrator`. The integration tests require the Docker stack:

```bash
docker compose up --build -d
pytest integration_tests
docker compose down
```

Set `GATEWAY_URL` when the gateway is running at a non-default address:

```bash
GATEWAY_URL=http://localhost:8000 pytest integration_tests
```

## Repository layout

```text
api-gateway/         Public FastAPI API and browser console
audio-engine/        WAV validation, feature extraction, and inference
agent-orchestrator/  Analyst, Cardiologist, and Auditor workflow
mlflow-server/       Local MLflow tracking and artifact service
integration_tests/   End-to-end pipeline tests
docker-compose.yml   Local multi-service deployment
```
