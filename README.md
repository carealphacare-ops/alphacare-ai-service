# AlphaCare — Python AI Service

NLP + ML-based symptom checker built with FastAPI and scikit-learn.

## Setup

```bash
cd ai-service
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Run

```bash
uvicorn main:app --reload --port 8000
```

## Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | /health | Service health check |
| POST | /analyze | Symptom analysis + disease prediction |
| GET | /symptoms | All known symptoms for autocomplete |
| GET | /specialties | Specialty key → label map |

## POST /analyze

**Request:**
```json
{ "symptoms": "I have a severe headache, nausea, and sensitivity to light for 2 days" }
```

**Response:**
```json
{
  "predictions": [
    { "name": "Migraine", "confidence": 87.3, "description": "...", "specKey": "neuro" },
    { "name": "Tension Headache", "confidence": 54.1, "description": "...", "specKey": "neuro" }
  ],
  "severity": "moderate",
  "isEmergency": false,
  "emergencyMessage": "",
  "recommendedSpecKey": "neuro",
  "recommendedSpecLabel": "Neurologist"
}
```
