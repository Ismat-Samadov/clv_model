from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from pydantic import BaseModel, Field, field_validator
from typing import List, Optional, Dict
import pandas as pd
import numpy as np
import uvicorn
import os
import logging
import time
from datetime import datetime
from model import BankingCLVModel
from fastapi.staticfiles import StaticFiles

# Configure logging
logging.basicConfig(
    filename='api.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Banking CLV Prediction API",
    description="API for predicting Customer Lifetime Value in banking",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Constants
MODEL_PATH = "models/clv_model_latest"
VALID_TRANSACTION_TYPES = ["Deposit", "Withdrawal", "Transfer", "Bill Payment", "Fee"]
VALID_CHANNELS = ["Online", "Mobile", "Branch", "ATM"]
VALID_PRODUCT_TYPES = ["Savings", "Checking", "Credit Card", "Mortgage", "Investment", "Personal Loan"]
VALID_REGIONS = ["North", "South", "East", "West"]
VALID_ACQUISITION_CHANNELS = ["Online", "Branch", "Referral", "Marketing"]
VALID_STATUS = ["Active", "Inactive", "Suspended"]

# Pydantic models with updated validation
class Transaction(BaseModel):
    transaction_date: str
    transaction_type: str = Field(..., description="Type of transaction")
    amount: float = Field(..., gt=-1000000, lt=1000000)
    channel: str

    @field_validator('transaction_type')
    @classmethod
    def validate_transaction_type(cls, v):
        if v not in VALID_TRANSACTION_TYPES:
            raise ValueError(f"Invalid transaction type. Must be one of {VALID_TRANSACTION_TYPES}")
        return v

    @field_validator('channel')
    @classmethod
    def validate_channel(cls, v):
        if v not in VALID_CHANNELS:
            raise ValueError(f"Invalid channel. Must be one of {VALID_CHANNELS}")
        return v

    @field_validator('transaction_date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

class Product(BaseModel):
    product_type: str = Field(..., description="Type of banking product")
    start_date: str
    balance: float = Field(..., ge=0)
    status: str = Field(..., description="Product status")

    @field_validator('status')
    @classmethod
    def validate_status(cls, v):
        if v not in VALID_STATUS:
            raise ValueError(f"Invalid status. Must be one of {VALID_STATUS}")
        return v

    @field_validator('product_type')
    @classmethod
    def validate_product_type(cls, v):
        if v not in VALID_PRODUCT_TYPES:
            raise ValueError(f"Invalid product type. Must be one of {VALID_PRODUCT_TYPES}")
        return v

    @field_validator('start_date')
    @classmethod
    def validate_date(cls, v):
        try:
            datetime.fromisoformat(v.replace('Z', '+00:00'))
            return v
        except ValueError:
            raise ValueError("Invalid date format. Use ISO format (YYYY-MM-DDTHH:MM:SS)")

class Customer(BaseModel):
    customer_id: int = Field(..., gt=0)
    age: int = Field(..., ge=18, le=120)
    income: float = Field(..., gt=0)
    credit_score: float = Field(..., ge=300, le=850)
    tenure_months: int = Field(..., ge=0)
    region: str
    acquisition_channel: str
    transactions: List[Transaction]
    products: List[Product]

    @field_validator('region')
    @classmethod
    def validate_region(cls, v):
        if v not in VALID_REGIONS:
            raise ValueError(f"Invalid region. Must be one of {VALID_REGIONS}")
        return v

    @field_validator('acquisition_channel')
    @classmethod
    def validate_acquisition_channel(cls, v):
        if v not in VALID_ACQUISITION_CHANNELS:
            raise ValueError(f"Invalid acquisition channel. Must be one of {VALID_ACQUISITION_CHANNELS}")
        return v

class PredictionResponse(BaseModel):
    customer_id: int
    predicted_clv: float
    prediction_timestamp: str
    confidence_score: float
    feature_importance: Optional[Dict[str, float]] = None

def calculate_confidence_score(transactions_df, products_df) -> float:
    """Calculate confidence score based on data quality."""
    base_score = 0.7
    
    # Add score for transactions
    if len(transactions_df) > 0:
        base_score += min(0.15, len(transactions_df) * 0.01)
        
    # Add score for products
    if len(products_df) > 0:
        base_score += min(0.1, len(products_df) * 0.03)
    
    return min(0.95, base_score)

# Global model instance
try:
    model = BankingCLVModel.load_model(MODEL_PATH)
    logger.info(f"Model loaded successfully from {MODEL_PATH}")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    model = None

# Dependency for model availability
async def get_model():
    if model is None:
        raise HTTPException(status_code=503, detail="Model not available")
    return model


@app.get("/", response_class=HTMLResponse)
async def root():
    with open("index.html", "r") as f:
        return f.read()
@app.post("/predict/clv", response_model=PredictionResponse)
async def predict_clv(customer: Customer, model: BankingCLVModel = Depends(get_model)):
    try:
        start_time = time.time()
        
        # Convert input data to DataFrames
        customer_df = pd.DataFrame([{
            'customer_id': customer.customer_id,
            'age': customer.age,
            'income': customer.income,
            'credit_score': customer.credit_score,
            'tenure_months': customer.tenure_months,
            'region': customer.region,
            'acquisition_channel': customer.acquisition_channel
        }])
        
        transactions_df = pd.DataFrame([
            {**t.dict(), 'customer_id': customer.customer_id}
            for t in customer.transactions
        ])
        transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
        
        products_df = pd.DataFrame([
            {**p.dict(), 'customer_id': customer.customer_id}
            for p in customer.products
        ])
        products_df['start_date'] = pd.to_datetime(products_df['start_date'])
        
        # Calculate metrics
        metrics_df = pd.DataFrame([{
            'customer_id': customer.customer_id,
            'total_transaction_amount': transactions_df['amount'].sum(),
            'transaction_count': len(transactions_df),
            'product_count': len(products_df),
            'avg_transaction_value': transactions_df['amount'].mean() if len(transactions_df) > 0 else 0,
            'total_balance': products_df['balance'].sum()
        }])
        
        # Make prediction
        prediction_result = model.predict(customer_df, transactions_df, products_df, metrics_df)
        
        # Calculate confidence score
        confidence_score = calculate_confidence_score(transactions_df, products_df)
        
        # Log prediction details
        process_time = time.time() - start_time
        logger.info(
            f"Prediction made for customer {customer.customer_id} "
            f"(CLV: ${prediction_result['predicted_clv'].iloc[0]:,.2f}, "
            f"Confidence: {confidence_score:.2f}) "
            f"in {process_time:.4f} seconds"
        )
        
        return {
            "customer_id": customer.customer_id,
            "predicted_clv": float(prediction_result['predicted_clv'].iloc[0]),
            "prediction_timestamp": datetime.now().isoformat(),
            "confidence_score": confidence_score,
            "feature_importance": None
        }
        
    except Exception as e:
        logger.error(f"Error in prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {
        "status": "healthy" if model is not None else "degraded",
        "model_version": os.path.basename(MODEL_PATH),
        "timestamp": datetime.now().isoformat(),
        "api_version": "1.0.0"
    }

# Error handler for unexpected errors
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Global error handler: {str(exc)}")
    return JSONResponse(
        status_code=500,
        content={"message": "An unexpected error occurred", "detail": str(exc)}
    )

# At the end of your script
if __name__ == "__main__":
    port = int(os.getenv("PORT", 8000))  # Use PORT env var, default to 8000
    uvicorn.run("api:app", host="0.0.0.0", port=port, reload=True)
