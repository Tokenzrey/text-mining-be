# DepressionEmo Classification API

An API for detecting depression-related emotions in text using a BERT-based deep learning model.

## Introduction

DepressionEmo Classification API provides a RESTful interface for analyzing text to detect 8 depression-related emotions:

- anger
- brain dysfunction (forget)
- emptiness
- hopelessness
- loneliness
- sadness
- suicide intent
- worthlessness

This API is based on the DepressionEmo dataset research and uses a fine-tuned BERT model for multilabel emotion classification.

## Installation

### Prerequisites

- Python 3.8+
- pip
- Virtual environment (recommended)

### Installation Steps

1. Clone the repository:

   ```bash
   git clone https://github.com/yourusername/depression-emotion-api.git
   cd depression-emotion-api
   ```

2. Create and activate a virtual environment:

   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Setup

### Model Files

The API requires pre-trained model files. Make sure you have:

1. Model weights file: `DepressionEmo_Models/bert_model.pt`
2. Tokenizer directory: `DepressionEmo_Models/bert_tokenizer/`

If you have the model files in a different location, you can specify them using environment variables:

```bash
export MODEL_PATH=/path/to/bert_model.pt
export TOKENIZER_PATH=/path/to/bert_tokenizer
```

## Running the API

### Local Development

Run the API locally using:

```bash
python app.py
```

The API will be available at http://localhost:8000.

### Production Deployment

For production deployment, we recommend using Gunicorn:

```bash
gunicorn app:app --workers 4 --worker-class uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000
```

### Using Docker

A Dockerfile is provided for containerized deployment:

```bash
# Build the Docker image
docker build -t depression-emotion-api .

# Run the container
docker run -p 8000:8000 depression-emotion-api
```

## API Documentation

### Endpoints

#### 1. Health Check

```
GET /health
```

Checks if the API and model are operational.

**Response:**

```json
{
	"status": "ok",
	"version": "1.0.0",
	"model_loaded": true
}
```

#### 2. Predict Emotions

```
POST /predict
```

Analyzes text for depression-related emotions.

**Request Body:**

```json
{
	"text": "I feel so empty and alone, nothing makes me happy anymore",
	"threshold": 0.5
}
```

| Parameter | Type   | Required | Description                                      |
| --------- | ------ | -------- | ------------------------------------------------ |
| text      | string | Yes      | Text to analyze (1-5000 chars)                   |
| threshold | float  | No       | Classification threshold (0.0-1.0, default: 0.5) |

**Response:**

```json
{
	"request_id": "550e8400-e29b-41d4-a716-446655440000",
	"emotions": ["emptiness", "loneliness", "sadness"],
	"probabilities": {
		"anger": 0.12,
		"brain dysfunction (forget)": 0.08,
		"emptiness": 0.85,
		"hopelessness": 0.45,
		"loneliness": 0.92,
		"sadness": 0.78,
		"suicide intent": 0.22,
		"worthlessness": 0.38
	},
	"processing_time_ms": 128.45,
	"version": "1.0.0"
}
```

### Using the API with cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "Content-Type: application/json" \
     -d '{
           "text": "I feel so empty and alone, nothing makes me happy anymore",
           "threshold": 0.5
         }'
```

### Using the API with Python

```python
import requests

url = "http://localhost:8000/predict"
data = {
    "text": "I feel so empty and alone, nothing makes me happy anymore",
    "threshold": 0.5
}

response = requests.post(url, json=data)
result = response.json()
print(result["emotions"])
```

## Model Information

The API uses a fine-tuned BERT model trained on the DepressionEmo dataset. The model was trained to detect 8 different depression-related emotions with the following performance metrics:

- Micro F1 Score: ~0.82
- Macro F1 Score: ~0.75

For more information about the model architecture and training process, see the research paper: [DepressionEmo: A novel dataset for multilabel classification of depression emotions](https://arxiv.org/pdf/2401.04655.pdf).

## Error Handling

The API returns appropriate HTTP status codes for different error scenarios:

- `400 Bad Request`: Invalid input parameters
- `500 Internal Server Error`: Server-side errors including model prediction failures
- `503 Service Unavailable`: Model not loaded or unavailable

Error responses include detailed information:

```json
{
	"error": "Internal Server Error",
	"detail": "Prediction failed: error details",
	"request_id": "550e8400-e29b-41d4-a716-446655440000"
}
```

## License

[Include license information here]
