from fastapi import FastAPI, HTTPException, Depends, Request, BackgroundTasks
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
import sys

# Configure logging
logging.basicConfig(
    level=logging.INFO, 
    format="%(asctime)s - %(name)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("depression_api.log")
    ]
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

# Global model service instance with thread safety
_model_service = None
_model_lock = threading.RLock()

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
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.bert = BertModel.from_pretrained(pretrained_model)
        self.drop = nn.Dropout(p=0.1)
        self.out = nn.Linear(self.bert.config.hidden_size, n_classes)
        logger.info(f"DEBUG: EmotionClassifier initialized successfully")

    def forward(self, input_ids, attention_mask):
        try:
            outputs = self.bert(
                input_ids=input_ids,
                attention_mask=attention_mask
            )
            pooled_output = outputs.pooler_output
            output = self.drop(pooled_output)
            return self.out(output)
        except Exception as e:
            logger.error(f"Forward pass error: {str(e)}")
            import traceback
            logger.error(traceback.format_exc())
            raise

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
        results = {
            "model_file_exists": False,
            "tokenizer_dir_exists": False,
            "model_file_size": 0,
            "tokenizer_files": []
        }
        
        if os.path.exists(self.model_path):
            results["model_file_exists"] = True
            results["model_file_size"] = os.path.getsize(self.model_path)
            logger.info(f"DEBUG: Model file exists with size: {results['model_file_size']} bytes")
        else:
            logger.error(f"DEBUG: Model file does not exist at: {self.model_path}")
        
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
        try:
            logger.info(f"DEBUG: Starting model loading process")
            
            path_check = self.check_paths()
            if not path_check["model_file_exists"]:
                raise FileNotFoundError(f"Model file not found: {self.model_path}")
            if not path_check["tokenizer_dir_exists"]:
                raise FileNotFoundError(f"Tokenizer directory not found: {self.tokenizer_path}")
                
            logger.info(f"DEBUG: Initializing model architecture")
            self.model = EmotionClassifier(len(EMOTION_LIST))
            
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
            
            try:
                logger.info(f"DEBUG: Loading tokenizer from {self.tokenizer_path}")
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings("ignore", category=FutureWarning)
                    self.tokenizer = BertTokenizer.from_pretrained(self.tokenizer_path)
                logger.info(f"DEBUG: Tokenizer loaded successfully")
                logger.info(f"DEBUG: Tokenizer vocabulary size: {len(self.tokenizer)}")
            except Exception as e:
                logger.error(f"DEBUG: Error loading tokenizer: {str(e)}")
                raise RuntimeError(f"Failed to load tokenizer: {str(e)}")
            
            self.is_loaded = True
            logger.info(f"DEBUG: Model and tokenizer loaded successfully")
            
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
            import traceback
            logger.error(traceback.format_exc())
            raise RuntimeError(f"Prediction failed: {str(e)}")

# Request ID generator
def generate_request_id() -> str:
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
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize model service as a dependency
def get_model_service() -> EmotionModelService:
    global _model_service, _model_lock
    
    logger.info(f"DEBUG: get_model_service called")
    
    with _model_lock:
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
        
        if not _model_service.is_loaded:
            logger.warning(f"DEBUG: Model not loaded, attempting to reload")
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

# API endpoints
@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check(request: Request, model_service: EmotionModelService = Depends(get_model_service)):
    request_id = generate_request_id()
    logger.info(f"DEBUG: Health check requested (ID: {request_id})")
    
    response = HealthResponse(
        status="ok",
        version=API_VERSION,
        model_loaded=model_service.is_loaded
    )
    
    logger.info(f"DEBUG: Health check response: {json.dumps(response.dict())}")
    return response

@app.post("/predict", response_model=PredictionResponse, tags=["Emotion Analysis"])
async def predict_emotions(
    prediction_request: PredictionRequest,
    background_tasks: BackgroundTasks,
    model_service: EmotionModelService = Depends(get_model_service)
):
    """
    Analyze text for depression-related emotions
    
    Returns a list of detected emotions and their probability scores
    """
    request_id = generate_request_id()
    start_time = time.time()
    
    # Log request details
    text_preview = prediction_request.text[:50] + "..." if len(prediction_request.text) > 50 else prediction_request.text
    logger.info(f"DEBUG: Prediction request received - ID: {request_id}, text: '{text_preview}'")
    logger.info(f"DEBUG: Using threshold: {prediction_request.threshold}")
    
    try:
        # Clean and prepare input text
        text = prediction_request.text.strip()
        
        if not text:
            logger.warning(f"DEBUG: Empty text provided in request {request_id}")
            raise HTTPException(status_code=400, detail="Empty text provided")
            
        # Get prediction - this should now work correctly
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
            request_id=request_id,
            emotions=prediction["emotions"],
            probabilities=prediction["probabilities"],
            processing_time_ms=processing_time,
            version=API_VERSION
        )
        
        # Log full response
        logger.info(f"DEBUG: Full prediction response prepared for request {request_id}")
        
        # Add logging task to background
        background_tasks.add_task(
            lambda: logger.info(f"DEBUG: Request {request_id} completed successfully")
        )
        
        return response
    except RuntimeError as e:
        logger.error(f"DEBUG: Prediction error in request {request_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )
    except Exception as e:
        logger.error(f"DEBUG: Unexpected error in request {request_id}: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail=f"An unexpected error occurred: {str(e)}"
        )

@app.get("/", tags=["System"])
async def root():
    """API root - provides basic information"""
    request_id = generate_request_id()
    logger.info(f"DEBUG: Root endpoint accessed (ID: {request_id})")
    return {
        "name": API_TITLE,
        "version": API_VERSION,
        "description": API_DESCRIPTION,
        "documentation": "/docs"
    }

def preload_model():
    """Preload model in background thread with thread safety"""
    logger.info("DEBUG: Preloading model in background")
    try:
        global _model_service, _model_lock
        with _model_lock:
            if _model_service is None:
                _model_service = EmotionModelService(MODEL_PATH, TOKENIZER_PATH)
                logger.info("DEBUG: Model preloaded successfully")
            else:
                logger.info("DEBUG: Model already loaded, skipping preload")
    except Exception as e:
        logger.error(f"DEBUG: Model preload failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())

# Enable this when running the file directly
if __name__ == "__main__":
    import uvicorn
    
    # Preload model in background thread
    threading.Thread(target=preload_model, daemon=True).start()
    
    port = 8000
    logger.info(f"DEBUG: Starting server on port {port}")
    uvicorn.run("app:app", host="0.0.0.0", port=port, reload=False)