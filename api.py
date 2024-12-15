from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, validator
from typing import Dict, List, Optional
from enum import Enum
import pandas as pd
import numpy as np
import joblib
import json
from pathlib import Path
import logging
import traceback

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
class CustomerProfile(BaseModel):
    # Basic Information
    address_state: str = Field(..., description="State of residence (e.g., CA, NY)")
    address_zip: int = Field(..., description="ZIP code")
    occupation: str = Field(..., description="Customer occupation")
    employer: str = Field(..., description="Employer name")
    income: float = Field(..., gt=0, description="Annual income")
    customer_segment: str = Field(..., description="Customer segment (MASS, AFFLUENT, PRIVATE, BUSINESS)")
    credit_score: float = Field(..., ge=300, le=850, description="Credit score")
    onboarding_branch_id: int = Field(..., description="Branch ID where customer was onboarded")
    
    # Account Counts
    checking_account_count: int = Field(..., ge=0, description="Number of checking accounts")
    savings_account_count: int = Field(..., ge=0, description="Number of savings accounts")
    credit_card_count: int = Field(..., ge=0, description="Number of credit cards")
    loan_count: int = Field(..., ge=0, description="Number of loans")
    checking_account_id_count: int = Field(..., ge=0)
    savings_account_id_count: int = Field(..., ge=0)
    credit_account_id_count: int = Field(..., ge=0)
    loan_account_id_count: int = Field(..., ge=0)
    total_accounts: int = Field(..., ge=0)
    product_diversity: int = Field(..., ge=0)
    
    # Balance Information
    total_balance: float = Field(..., ge=0, description="Total balance across all accounts")
    checking_current_balance_mean: float = Field(..., ge=0)
    checking_current_balance_sum: float = Field(..., ge=0)
    checking_current_balance_std: float = Field(..., ge=0)
    savings_current_balance_mean: float = Field(..., ge=0)
    savings_current_balance_sum: float = Field(..., ge=0)
    savings_current_balance_std: float = Field(..., ge=0)
    credit_current_balance_mean: float = Field(..., ge=0)
    credit_current_balance_sum: float = Field(..., ge=0)
    credit_current_balance_std: float = Field(..., ge=0)
    loan_current_balance_mean: float = Field(..., ge=0)
    loan_current_balance_sum: float = Field(..., ge=0)
    loan_current_balance_std: float = Field(..., ge=0)
    loan_total_original: float = Field(..., ge=0)
    
    # Transaction Information
    avg_monthly_transactions: float = Field(..., ge=0, description="Average monthly transaction amount")
    checking_tx_amount_mean: float = Field(..., ge=0)
    checking_tx_amount_sum: float = Field(..., ge=0)
    checking_tx_amount_std: float = Field(..., ge=0)
    checking_tx_amount_count: int = Field(..., ge=0)
    checking_tx_transaction_id_nunique: int = Field(..., ge=0)
    
    savings_tx_amount_mean: float = Field(..., ge=0)
    savings_tx_amount_sum: float = Field(..., ge=0)
    savings_tx_amount_std: float = Field(..., ge=0)
    savings_tx_amount_count: int = Field(..., ge=0)
    savings_tx_transaction_id_nunique: int = Field(..., ge=0)
    
    credit_tx_amount_mean: float = Field(..., ge=0)
    credit_tx_amount_sum: float = Field(..., ge=0)
    credit_tx_amount_std: float = Field(..., ge=0)
    credit_tx_amount_count: int = Field(..., ge=0)
    credit_tx_transaction_id_nunique: int = Field(..., ge=0)
    
    loan_tx_amount_mean: float = Field(..., ge=0)
    loan_tx_amount_sum: float = Field(..., ge=0)
    loan_tx_amount_std: float = Field(..., ge=0)
    loan_tx_amount_count: int = Field(..., ge=0)
    loan_tx_transaction_id_nunique: int = Field(..., ge=0)
    
    # Risk Metrics
    credit_score_x: float = Field(..., ge=300, le=850)
    credit_score_y: float = Field(..., ge=300, le=850)
    credit_utilization: float = Field(..., ge=0, le=1)
    credit_utilization_mean: float = Field(..., ge=0, le=1)
    debt_to_income: float = Field(..., ge=0)
    payment_history_score: float = Field(..., ge=0, le=1)
    missed_payments_count: int = Field(..., ge=0)
    risk_score: float = Field(..., ge=300, le=850)
    
    # Merchant Information
    merchant_merchant_id_nunique: int = Field(..., ge=0)
    merchant_category_nunique: int = Field(..., ge=0)
    merchant_amount_mean: float = Field(..., ge=0)
    merchant_amount_sum: float = Field(..., ge=0)
    merchant_amount_std: float = Field(..., ge=0)

    @validator('customer_segment')
    def validate_customer_segment(cls, v):
        allowed_segments = ['MASS', 'AFFLUENT', 'PRIVATE', 'BUSINESS']
        if v not in allowed_segments:
            raise ValueError(f"Customer segment must be one of {allowed_segments}")
        return v
    
class PredictionResponse(BaseModel):
    risk_category: str
    probability: Dict[str, float]
    model_confidence: float

class TestProfile(str, Enum):
    EXCELLENT = "excellent"
    VERY_LOW_RISK = "very_low_risk"
    LOW_RISK = "low_risk"
    MEDIUM_RISK_STABLE = "medium_risk_stable"
    MEDIUM_RISK_NEW = "medium_risk_new"
    HIGH_RISK_EMPLOYED = "high_risk_employed"
    HIGH_RISK_STUDENT = "high_risk_student"
    VERY_HIGH_RISK = "very_high_risk"
    BUSINESS = "business"
    RETIRED = "retired"

# Test profiles data
TEST_PROFILES = {
    TestProfile.EXCELLENT: {
        "address_state": "NY",
        "address_zip": 10021,
        "occupation": "CEO",
        "employer": "Fortune 500 Corp",
        "income": 500000,
        "customer_segment": "PRIVATE",
        "credit_score": 850,
        "checking_account_count": 3,
        "savings_account_count": 4,
        "credit_card_count": 5,
        "loan_count": 1,
        "total_balance": 2500000,
        "avg_monthly_transactions": 25000
    },
    TestProfile.VERY_LOW_RISK: {
        "address_state": "CA",
        "address_zip": 94105,
        "occupation": "Senior Software Engineer",
        "employer": "Tech Giant Inc",
        "income": 250000,
        "customer_segment": "AFFLUENT",
        "credit_score": 820,
        "checking_account_count": 2,
        "savings_account_count": 2,
        "credit_card_count": 3,
        "loan_count": 1,
        "total_balance": 150000,
        "avg_monthly_transactions": 8000
    },
    TestProfile.LOW_RISK: {
        "address_state": "MA",
        "address_zip": 2110,
        "occupation": "Doctor",
        "employer": "City Hospital",
        "income": 180000,
        "customer_segment": "AFFLUENT",
        "credit_score": 780,
        "checking_account_count": 2,
        "savings_account_count": 1,
        "credit_card_count": 2,
        "loan_count": 1,
        "total_balance": 85000,
        "avg_monthly_transactions": 6000
    },
    TestProfile.MEDIUM_RISK_STABLE: {
        "address_state": "TX",
        "address_zip": 75001,
        "occupation": "Store Manager",
        "employer": "Retail Chain Corp",
        "income": 65000,
        "customer_segment": "MASS",
        "credit_score": 720,
        "checking_account_count": 1,
        "savings_account_count": 1,
        "credit_card_count": 2,
        "loan_count": 1,
        "total_balance": 15000,
        "avg_monthly_transactions": 3000
    },
    TestProfile.MEDIUM_RISK_NEW: {
        "address_state": "IL",
        "address_zip": 60601,
        "occupation": "Sales Representative",
        "employer": "StartUp Inc",
        "income": 55000,
        "customer_segment": "MASS",
        "credit_score": 680,
        "checking_account_count": 1,
        "savings_account_count": 0,
        "credit_card_count": 1,
        "loan_count": 0,
        "total_balance": 5000,
        "avg_monthly_transactions": 2500
    },
    TestProfile.HIGH_RISK_EMPLOYED: {
        "address_state": "FL",
        "address_zip": 33101,
        "occupation": "Gig Worker",
        "employer": "Self Employed",
        "income": 35000,
        "customer_segment": "MASS",
        "credit_score": 600,
        "checking_account_count": 1,
        "savings_account_count": 0,
        "credit_card_count": 1,
        "loan_count": 2,
        "total_balance": 2000,
        "avg_monthly_transactions": 1500
    },
    TestProfile.HIGH_RISK_STUDENT: {
        "address_state": "CA",
        "address_zip": 90024,
        "occupation": "Student",
        "employer": "UCLA",
        "income": 15000,
        "customer_segment": "MASS",
        "credit_score": 580,
        "checking_account_count": 1,
        "savings_account_count": 0,
        "credit_card_count": 1,
        "loan_count": 2,
        "total_balance": 1000,
        "avg_monthly_transactions": 800
    },
    TestProfile.VERY_HIGH_RISK: {
        "address_state": "NV",
        "address_zip": 89101,
        "occupation": "Unemployed",
        "employer": "None",
        "income": 12000,
        "customer_segment": "MASS",
        "credit_score": 520,
        "checking_account_count": 1,
        "savings_account_count": 0,
        "credit_card_count": 0,
        "loan_count": 3,
        "total_balance": 500,
        "avg_monthly_transactions": 600
    },
    TestProfile.BUSINESS: {
        "address_state": "WA",
        "address_zip": 98101,
        "occupation": "Business Owner",
        "employer": "Small Business LLC",
        "income": 120000,
        "customer_segment": "BUSINESS",
        "credit_score": 750,
        "checking_account_count": 2,
        "savings_account_count": 1,
        "credit_card_count": 2,
        "loan_count": 1,
        "total_balance": 50000,
        "avg_monthly_transactions": 15000
    },
    TestProfile.RETIRED: {
        "address_state": "AZ",
        "address_zip": 85001,
        "occupation": "Retired",
        "employer": "Retired",
        "income": 75000,
        "customer_segment": "MASS",
        "credit_score": 760,
        "checking_account_count": 1,
        "savings_account_count": 2,
        "credit_card_count": 1,
        "loan_count": 0,
        "total_balance": 200000,
        "avg_monthly_transactions": 4000
    }
}

PROFILE_DESCRIPTIONS = {
    TestProfile.EXCELLENT: "High net worth individual with perfect credit history and substantial assets",
    TestProfile.VERY_LOW_RISK: "High-income tech professional with excellent credit and stable financial history",
    TestProfile.LOW_RISK: "Healthcare professional with strong credit and good financial management",
    TestProfile.MEDIUM_RISK_STABLE: "Stable retail manager with moderate income and good credit",
    TestProfile.MEDIUM_RISK_NEW: "Early career professional with limited credit history",
    TestProfile.HIGH_RISK_EMPLOYED: "Gig economy worker with variable income and credit challenges",
    TestProfile.HIGH_RISK_STUDENT: "Student with limited income and significant student loans",
    TestProfile.VERY_HIGH_RISK: "Unemployed individual with poor credit history",
    TestProfile.BUSINESS: "Small business owner with mixed personal and business finances",
    TestProfile.RETIRED: "Retired individual with fixed income and substantial savings"
}

# Initialize FastAPI app
app = FastAPI(
    title="Banking Risk Assessment API",
    description="API for assessing customer risk based on banking behavior",
    version="1.0.0"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
static_dir = Path("static")
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Global variables for model artifacts
model = None
scaler = None
label_encoders = None
feature_names = None
model_evaluation = None

def load_model_artifacts():
    """Load all model artifacts"""
    global model, scaler, label_encoders, feature_names, model_evaluation
    
    MODEL_DIR = Path("trained_models/models")
    
    try:
        model = joblib.load(MODEL_DIR / "best_model.joblib")
        logger.info("Model loaded successfully")
        
        scaler = joblib.load(MODEL_DIR / "scaler.joblib")
        logger.info("Scaler loaded successfully")
        
        label_encoders = joblib.load(MODEL_DIR / "label_encoders.joblib")
        logger.info("Label encoders loaded successfully")

        with open(MODEL_DIR / "feature_names.json", "r") as f:
            feature_names = json.load(f)
        logger.info("Feature names loaded successfully")

        with open(MODEL_DIR / "evaluation_results.json", "r") as f:
            model_evaluation = json.load(f)
        logger.info("Model evaluation results loaded successfully")
        
        return True
    except Exception as e:
        logger.error(f"Error loading model artifacts: {str(e)}")
        logger.error(traceback.format_exc())
        return False

# Load model artifacts on startup
@app.on_event("startup")
async def startup_event():
    logger.info("Starting up the application...")
    if not load_model_artifacts():
        logger.error("Failed to load model artifacts during startup")

@app.get("/")
async def root():
    """Serve the frontend HTML"""
    try:
        return FileResponse('static/index.html')
    except Exception as e:
        logger.error(f"Error serving index.html: {str(e)}")
        raise HTTPException(status_code=500, detail="Error serving frontend")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if all([model, scaler, label_encoders]) else "unhealthy",
        "model_loaded": model is not None,
        "scaler_loaded": scaler is not None,
        "encoders_loaded": label_encoders is not None,
        "feature_names_loaded": feature_names is not None,
        "evaluation_loaded": model_evaluation is not None
    }

@app.get("/test-profiles")
async def get_test_profiles():
    """Get list of available test profiles with descriptions"""
    return {
        "profiles": [
            {
                "id": profile.value,
                "name": profile.value.replace("_", " ").title(),
                "description": PROFILE_DESCRIPTIONS[profile]
            }
            for profile in TestProfile
        ]
    }

@app.get("/test-profiles/{profile_id}")
async def get_test_profile(profile_id: TestProfile):
    """Get specific test profile data"""
    return TEST_PROFILES[profile_id]

@app.post("/predict", response_model=PredictionResponse)
async def predict(customer_data: CustomerProfile):
    """Make risk prediction for customer"""
    if not all([model, scaler, label_encoders]):
        raise HTTPException(
            status_code=503,
            detail="Model not properly loaded"
        )

    try:
        logger.info("Processing prediction request")
        
        # Convert input data to DataFrame
        data = pd.DataFrame([customer_data.dict()])
        logger.debug(f"Input data shape: {data.shape}")
        
        # Verify features
        missing_features = set(feature_names) - set(data.columns)
        if missing_features:
            raise HTTPException(
                status_code=400,
                detail=f"Missing features: {missing_features}"
            )

        # Encode categorical variables
        for col in data.select_dtypes(include=['object']).columns:
            if col in label_encoders:
                try:
                    data[col] = label_encoders[col].transform(data[col].astype(str))
                except Exception as e:
                    logger.error(f"Error encoding {col}: {str(e)}")
                    raise HTTPException(
                        status_code=400,
                        detail=f"Invalid value in {col}"
                    )

        # Scale features
        try:
            data_scaled = pd.DataFrame(
                scaler.transform(data),
                columns=data.columns
            )
        except Exception as e:
            logger.error(f"Error scaling features: {str(e)}")
            raise HTTPException(
                status_code=400,
                detail="Error scaling features"
            )

        # Make prediction
        try:
            risk_category = model.predict(data_scaled)[0]
            probabilities = model.predict_proba(data_scaled)[0]
            
            classes = model.classes_
            prob_dict = {
                str(class_): float(prob) 
                for class_, prob in zip(classes, probabilities)
            }
            
            confidence = float(max(probabilities))
            
            logger.info(f"Prediction successful: {risk_category}")
            
            return PredictionResponse(
                risk_category=str(risk_category),
                probability=prob_dict,
                model_confidence=confidence
            )
        except Exception as e:
            logger.error(f"Error making prediction: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail="Error making prediction"
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        logger.error(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail="An unexpected error occurred"
        )

# Error handling for validation errors
@app.exception_handler(ValueError)
async def validation_exception_handler(request, exc):
    logger.error(f"Validation error: {str(exc)}")
    return JSONResponse(
        status_code=400,
        content={"detail": str(exc)}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)