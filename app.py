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
import json
import threading

# Initialize only once to prevent duplicate logging
if not hasattr(globals(), "_INITIALIZED"):
    # Configure logging with detailed format
    logging.basicConfig(
        level=logging.INFO, 
        format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s"
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

    # Debug print for environment
    logger.info(f"DEBUG: Environment configuration:")
    logger.info(f"DEBUG: MODEL_PATH = {MODEL_PATH}")
    logger.info(f"DEBUG: TOKENIZER_PATH = {TOKENIZER_PATH}")
    logger.info(f"DEBUG: DEVICE = {DEVICE}")
    logger.info(f"DEBUG: PyTorch version = {torch.__version__}")
    logger.info(f"DEBUG: CUDA available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        logger.info(f"DEBUG: CUDA version = {torch.version.cuda}")
        logger.info(f"DEBUG: GPU = {torch.cuda.get_device_name(0)}")
        
    # Mark as initialized to prevent duplicate execution
    globals()["_INITIALIZED"] = True

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

logger.info(f"DEBUG: Emotions to detect: {EMOTION_LIST}")

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
    
    # Fix for Pydantic warning by disabling protected namespaces
    model_config = {
        "protected_namespaces": ()
    }
    
class ErrorResponse(BaseModel):
    error: str
    detail: Optional[str] = None
    request_id: str

# Model definition
class EmotionClassifier(nn.Module):
    def __init__(self, n_classes, pretrained_model='bert-base-cased'):
        super(EmotionClassifier, self).__init__()
        logger.info(f"DEBUG: Initializing EmotionClassifier with {n_classes} classes")
        self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        logger.info(f"DEBUG: EmotionClassifier initialized successfully")

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
        logger.info(f"DEBUG: Initializing EmotionModelService")
        logger.info(f"DEBUG: Model path: {model_path}")
        logger.info(f"DEBUG: Tokenizer path: {tokenizer_path}")
        self.tokenizer = None
        self.model = None
        self.model_path = model_path
        self.tokenizer_path = tokenizer_path
        self.is_loaded = False
        self.load_model()

    def check_paths(self) -> Dict[str, bool]:
        """Validate that model and tokenizer paths exist with detailed information"""
        results = {
            "model_file_exists": False,
            "tokenizer_dir_exists": False,
            "model_file_size": 0,
            "tokenizer_files": []
        }
        
        # Check model file
        if os.path.exists(self.model_path):
            results["model_file_exists"] = True
            results["model_file_size"] = os.path.getsize(self.model_path)
            logger.info(f"DEBUG: Model file exists with size: {results['model_file_size']} bytes")
        else:
            logger.error(f"DEBUG: Model file does not exist at: {self.model_path}")
        
        # Check tokenizer directory
        if os.path.exists(self.tokenizer_path):
            results["tokenizer_dir_exists"] = True
            if os.path.isdir(self.tokenizer_path):
                tokenizer_files = os.listdir(self.tokenizer_path)
                results["tokenizer_files"] = tokenizer_files
                logger.info(f"DEBUG: Tokenizer directory exists with files: {tokenizer_files}")
            else:
                logger.error(f"DEBUG: Tokenizer path exists but is not a directory: {self.tokenizer_path}")
        else:
            logger.error(f"DEBUG: Tokenizer directory does not exist at: {self.tokenizer_path}")
        
        return results

    def load_model(self) -> None:
        """Load the BERT model and tokenizer with detailed error reporting"""
        try:
            logger.info(f"DEBUG: Starting model loading process")
            
            # Detailed path checking
            path_check = self.check_paths()
            if not path_check["model_file_exists"]:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not path_check["tokenizer_dir_exists"]:
                raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")
                
            # Initialize model
            logger.info(f"DEBUG: Initializing model architecture")
            self.model = EmotionClassifier(len(EMOTION_LIST))
            
            # Load model weights with detailed error handling
            try:
                logger.info(f"DEBUG: Loading model weights from {self.model_path}")
                state_dict = torch.load(self.model_path, map_location=DEVICE)
                logger.info(f"DEBUG: Model state dict loaded, keys: {list(state_dict.keys())[:5]}...")
                self.model.load_state_dict(state_dict)
                logger.info(f"DEBUG: Model weights loaded successfully")
            except Exception as e:
                logger.error(f"DEBUG: Error loading model weights: {str(e)}")
                raise RuntimeError(f"Failed to load model weights: {str(e)}")
                
            self.model.to(DEVICE)
            logger.info(f"DEBUG: Model moved to device: {DEVICE}")
            self.model.eval()
            logger.info(f"DEBUG: Model set to evaluation mode")
            
            # Load tokenizer with detailed error handling
            try:
                logger.info(f"DEBUG: Loading tokenizer from {self.tokenizer_path}")
                self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
                logger.info(f"DEBUG: Tokenizer loaded successfully")
                logger.info(f"DEBUG: Tokenizer vocabulary size: {len(self.tokenizer)}")
            except Exception as e:
                logger.error(f"DEBUG: Error loading tokenizer: {str(e)}")
                raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
            
            self.is_loaded = True
            logger.info(f"DEBUG: Model and tokenizer loaded successfully")
            
            # Test tokenization to validate tokenizer
            try:
                sample_text = "This is a test."
                encoded = self.tokenizer(sample_text, return_tensors='pt')
                logger.info(f"DEBUG: Tokenizer test successful, tokens: {encoded['input_ids'].shape}")
            except Exception as e:
                logger.error(f"DEBUG: Tokenizer test failed: {str(e)}")
                
        except Exception as e:
            logger.error(f"DEBUG: Error in load_model: {str(e)}")
            self.is_loaded = False
            raise RuntimeError(f"Failed to load model: {str(e)}")

    def predict(self, text: str, threshold: float = 0.5) -> dict:
        """
        Process text and return prediction with detailed logging
        """
        if not self.is_loaded:
            logger.error("DEBUG: Prediction attempted but model not loaded")
            raise RuntimeError("Model not loaded")
            
        try:
            logger.info(f"DEBUG: Starting prediction for text of length {len(text)}")
            
            # Tokenize input text
            logger.info(f"DEBUG: Tokenizing input text")
            encoded_text = self.tokenizer(
                text,
                add_special_tokens=True,
                max_length=MAX_LENGTH,
                padding='max_length',
                truncation=True,
                return_tensors='pt'
            )
            logger.info(f"DEBUG: Tokenized text shape: {encoded_text['input_ids'].shape}")

            # Move tensors to the correct device
            input_ids = encoded_text['input_ids'].to(DEVICE)
            attention_mask = encoded_text['attention_mask'].to(DEVICE)
            logger.info(f"DEBUG: Tensors moved to device {DEVICE}")

            # Make prediction
            logger.info(f"DEBUG: Running model inference")
            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
                logger.info(f"DEBUG: Raw output shape: {outputs.shape}")
                probabilities = torch.sigmoid(outputs).cpu().numpy()[0]
                logger.info(f"DEBUG: Probabilities calculated, shape: {probabilities.shape}")

            # Convert predictions to emotions and probabilities
            emotions = [EMOTION_LIST[i] for i, prob in enumerate(probabilities) if prob >= threshold]
            prob_dict = {emotion: float(probabilities[i]) for i, emotion in enumerate(EMOTION_LIST)}
            
            logger.info(f"DEBUG: Detected emotions: {emotions}")
            logger.info(f"DEBUG: Highest probability: {max(probabilities):.4f}")
            
            result = {
                "emotions": emotions,
                "probabilities": prob_dict
            }

            return result
        except Exception as e:
            logger.error(f"DEBUG: Prediction error: {str(e)}")
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Request ID generator
def generate_request_id() -> str:
    """Generate a unique request ID"""
    req_id = str(uuid.uuid4())
    logger.info(f"DEBUG: Generated request ID: {req_id}")
    return req_id

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

logger.info(f"DEBUG: FastAPI app initialized with CORS middleware")

# Initialize model service as a dependency
def get_model_service() -> EmotionModelService:
    """Dependency for the model service with detailed error logs"""
    global _model_service
    
    logger.info(f"DEBUG: get_model_service called")
    
    # Create model service if it doesn't exist
    if _model_service is None:
        try:
            logger.info(f"DEBUG: Model service not initialized, creating new instance")
            _model_service = EmotionModelService(MODEL_PATH, TOKENIZER_PATH)
        except Exception as e:
            logger.error(f"DEBUG: Failed to initialize model service: {str(e)}")
            raise HTTPException(
                status_code=500, 
                detail=f"Model service initialization failed: {str(e)}"
            )
    
    # Verify model is loaded
    if not _model_service.is_loaded:
        logger.warning(f"DEBUG: Model not loaded, attempting to reload")
        # Try to reload the model
        try:
            logger.info("DEBUG: Attempting to reload model...")
            _model_service.load_model()
        except Exception as e:
            logger.error(f"DEBUG: Model reload failed: {str(e)}")
            raise HTTPException(
                status_code=503, 
                detail=f"Model service unavailable: {str(e)}"
            )
    
    logger.info(f"DEBUG: Model service ready")
    return _model_service

# Process request and handle errors
async def process_request(request: Request):
    """Process incoming request and attach unique ID"""
    request.state.request_id = generate_request_id()
    request.state.start_time = time.time()
    logger.info(f"DEBUG: Request processed with ID: {request.state.request_id}")

# Register middleware
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log request details and handle errors with detailed logs"""
    await process_request(request)
    logger.info(f"DEBUG: Request {request.state.request_id}: {request.method} {request.url.path}")
    
    try:
        # Get request body for debugging (limited size)
        body = await request.body()
        if len(body) > 0:
            try:
                # Try to decode as JSON
                body_text = body.decode('utf-8')
                if len(body_text) > 500:
                    body_text = body_text[:500] + "... [truncated]"
                logger.info(f"DEBUG: Request body: {body_text}")
            except Exception as e:
                logger.info(f"DEBUG: Could not decode request body: {str(e)}")
        
        response = await call_next(request)
        process_time = time.time() - request.state.start_time
        logger.info(f"DEBUG: Request {request.state.request_id} completed in {process_time:.3f}s with status {response.status_code}")
        
        return response
    except Exception as e:
        logger.error(f"DEBUG: Request {request.state.request_id} failed: {str(e)}")
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
async def health_check(request: Request, model_service: EmotionModelService = Depends(get_model_service)):
    """Check API and model health"""
    logger.info(f"DEBUG: Health check requested (ID: {request.state.request_id})")
    
    # Enhanced health check with more details
    response = HealthResponse(
        status="ok",
        version=API_VERSION,
        model_loaded=model_service.is_loaded
    )
    
    logger.info(f"DEBUG: Health check response: {json.dumps(response.dict())}")
    return response

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
    
    # Log request details
    text_preview = prediction_request.text[:50] + "..." if len(prediction_request.text) > 50 else prediction_request.text
    logger.info(f"DEBUG: Prediction request received - ID: {request.state.request_id}, text: '{text_preview}'")
    logger.info(f"DEBUG: Using threshold: {prediction_request.threshold}")
    
    try:
        # Clean and prepare input text
        text = prediction_request.text.strip()
        
        if not text:
            logger.warning(f"DEBUG: Empty text provided in request {request.state.request_id}")
            raise HTTPException(status_code=400, detail="Empty text provided")
            
        # Get prediction
        logger.info(f"DEBUG: Sending text to model for prediction")
        prediction = model_service.predict(
            text,
            prediction_request.threshold
        )
        
        # Calculate processing time
        processing_time = (time.time() - start_time) * 1000  # Convert to milliseconds
        logger.info(f"DEBUG: Prediction completed in {processing_time:.2f}ms")
        
        # Log prediction results
        logger.info(f"DEBUG: Detected emotions: {prediction['emotions']}")
        logger.info(f"DEBUG: Top emotions with probabilities: " + 
                  ", ".join([f"{e}: {prediction['probabilities'][e]:.4f}" 
                           for e in sorted(prediction['probabilities'], 
                                         key=lambda x: prediction['probabilities'][x], 
                                         reverse=True)[:3]]))
        
        # Create response
        response = PredictionResponse(
            request_id=request.state.request_id,
            emotions=prediction["emotions"],
            probabilities=prediction["probabilities"],
            processing_time_ms=processing_time,
            version=API_VERSION
        )
        
        return response
    except RuntimeError as e:
        logger.error(f"DEBUG: Prediction error in request {request.state.request_id}: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"DEBUG: Unexpected error in request {request.state.request_id}: {str(e)}")
        logger.exception("Detailed exception information:")
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

# Root endpoint for basic info
@app.get("/", tags=["System"])
async def root(request: Request):
    """API root - provides basic information"""
    logger.info(f"DEBUG: Root endpoint accessed (ID: {request.state.request_id})")
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "documentation": "/docs"
    }

def preload_model():
    """Preload model in background thread"""
    logger.info("DEBUG: Preloading model in background")
    try:
        global _model_service
        _model_service = EmotionModelService(MODEL_PATH, TOKENIZER_PATH)
        logger.info("DEBUG: Model preloaded successfully")
    except Exception as e:
        logger.error(f"DEBUG: Model preload failed: {str(e)}")

# Enable this when running the file directly
if __name__ == "__main__":
    import uvicorn
    
    # Preload model in background thread
    threading.Thread(target=preload_model, daemon=True).start()
    
    # Set port to 8000 as requested
    port = int(os.environ.get("PORT", 8000))
    
    logger.info(f"DEBUG: Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)