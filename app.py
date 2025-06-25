from fastapi import FastAPI, HTTPException, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import torch
from transformers import BertTokenizer, BertModel
from torch import nn
import logging
import time
import os
import uuid

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("depression-emotion-api")

# API metadata
API_VERSION = "1.0.0"
API_TITLE = "DepressionEmo Classification API"
API_DESCRIPTION = "API for detecting depression-related emotions in text"

# Constants
MODEL_PATH = os.environ.get("MODEL_PATH", "DepressionEmo_Models/bert_model.pt")
TOKENIZER_PATH = os.environ.get("TOKENIZER_PATH", "DepressionEmo_Models/bert_tokenizer")
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# List of emotions to detect
EMOTION_LIST = [
    'anger', 
    'brain dysfunction (forget)', 
    'emptiness', 
    'hopelessness',
    'loneliness', 
    'sadness', 
    'suicide intent', 
    'worthlessness'
]

# Global model service instance
_model_service = None

# Define request and response models
class PredictionRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=5000, description="Text to analyze for depression emotions")
    threshold: float = Field(0.5, ge=0.0, le=1.0, description="Classification threshold")

class PredictionResponse(BaseModel):
    request_id: str
    emotions: List[str] = Field(..., description="List of detected emotions")
    probabilities: Dict[str, float] = Field(..., description="Probability score for each emotion")
    processing_time_ms: float
    version: str = API_VERSION

class HealthResponse(BaseModel):
    status: str
    version: str
    model_loaded: bool
    
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: str

# Model definition
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model='bert-base-cased'):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled_output = outputs.pooler_output
        output = self.drop(pooled_output)
        return self.out(output)

# Model service class
class EmotionModelService:
    def __init__(self, model_path: str, tokenizer_path: str):
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_loaded = False
        self.load_model()

    def load_model(self) -> None:
        """Load the BERT model and tokenizer"""
        try:
            logger.info(f"Loading model from {self.model_path}")
            
            # Check if model file exists
            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            
            # Check if tokenizer directory exists
            if not os.path.exists(self.tokenizer_path):
                raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")
                
            # Initialize model
            self.model = EmotionClassifier(len(EMOTION_LIST))
            
            # Load model weights
            self.model.load_state_dict(torch.load(
                self.model_path, 
                map_location=DEVICE
            ))
            self.model.to(DEVICE)
            self.model.eval()
            
            # Load tokenizer
            logger.info(f"Loading tokenizer from {self.tokenizer_path}")
            self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
            
            self.is_loaded = True
            logger.info(f"Model and tokenizer loaded successfully on {DEVICE}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            self.is_loaded = False
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        """
        Process text and return prediction

        Args:
            text: Input text to analyze
            threshold: Threshold for classification (0 to 1)

        Returns:
            Dictionary with emotions and probabilities
        """
        if not self.is_loaded:
            logger.error("Model not loaded")
            raise RuntimeError("Model not loaded")
            
        try:
            # Tokenize input text
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )

            # Move tensors to the correct device
            input_ids = encoded_text['input_ids'].to(DEVICE)
            attention_mask = encoded_text['attention_mask'].to(DEVICE)

            # Make prediction
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]

            # Convert predictions to emotions and probabilities
            result = {
                "emotions": [EMOTION_LIST[i] for i, prob in enumerate(probabilities) if prob >= threshold],
                "probabilities": {emotion: float(probabilities[i]) for i, emotion in enumerate(EMOTION_LIST)}
            }

            return result
        except Exception as e:
            logger.error(f"Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Request ID generator
def generate_request_id() -> str:
    """Generate a unique request ID"""
    return str(uuid.uuid4())

# Initialize FastAPI app
app = FastAPI(
    title=API_TITLE,
    description=API_DESCRIPTION,
    version=API_VERSION,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

# Initialize model service as a dependency
def get_model_service() -> EmotionModelService:
    """Dependency for the model service"""
    global _model_service
    
    # Create model service if it doesn't exist
    if _model_service is None:
        try:
            _model_service = EmotionModelService(MODEL_PATH, TOKENIZER_PATH)
        except Exception as e:
            logger.error(f"Failed to initialize model service: {str(e)}")
            raise HTTPException(status_code=500, detail=f"Model service initialization failed: {str(e)}")
    
    # Verify model is loaded
    if not _model_service.is_loaded:
        # Try to reload the model
        try:
            logger.info("Attempting to reload model...")
            _model_service.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail="Model service unavailable")
    
    return _model_service

# Process request and handle errors
async def process_request(request: Request):
    """Process incoming request and attach unique ID"""
    request.state.request_id = generate_request_id()
    request.state.start_time = time.time()

# Register middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request details and handle errors"""
    await process_request(request)
    logger.info(f"Request {request.state.request_id}: {request.method} {request.url.path}")
    
    try:
        response = await call_next(request)
        process_time = time.time() - request.state.start_time
        logger.info(f"Request {request.state.request_id} completed in {process_time:.3f}s")
        return response
    except Exception as e:
        logger.error(f"Request {request.state.request_id} failed: {str(e)}")
        error_response = ErrorResponse(
            error="Internal Server Error",
            detail=str(e),
            request_id=request.state.request_id
        )
        return JSONResponse(
            status_code=500,
            content=error_response.dict()
        )

# API endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(model_service: EmotionModelService = Depends(get_model_service)):
    """Check API and model health"""
    return HealthResponse(
        status="ok",
        version=API_VERSION,
        model_loaded=model_service.is_loaded
    )

@app.post("/predict", response_model=PredictionResponse, tags=["Emotion Analysis"])
async def predict_emotions(
    request: Request,
    prediction_request: PredictionRequest,
    model_service: EmotionModelService = Depends(get_model_service)
):
    """
    Analyze text for depression-related emotions
    
    Returns a list of detected emotions and their probability scores
    """
    start_time = time.time()
    
    try:
        # Clean and prepare input text
        text = prediction_request.text.strip()
        
        if not text:
            raise HTTPException(status_code=400, detail="Empty text provided")
            
        # Get prediction
        prediction = model_service.predict(
            text,
            prediction_request.threshold
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Create response
        return PredictionResponse(
            request_id=request.state.request_id,
            emotions=prediction["emotions"],
            probabilities=prediction["probabilities"],
            processing_time_ms=processing_time,
            version=API_VERSION
        )
    except RuntimeError as e:
        logger.error(f"Prediction error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Root endpoint for basic info
@app.get("/", tags=["System"])
async def root():
    """API root - provides basic information"""
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "documentation": "/docs"
    }

# Enable this when running the file directly
if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)