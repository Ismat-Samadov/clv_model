# Banking Customer Lifetime Value (CLV) Prediction Model

This project implements a machine learning system for predicting Customer Lifetime Value in banking, including data generation, model training, and a web-based prediction API.

## Project Overview

The system consists of three main components:
1. Synthetic data generation for banking customers
2. Machine learning model for CLV prediction
3. FastAPI-based web service with an interactive UI

### Directory Structure
```
.
├── api.py              # FastAPI web service implementation
├── data_gen.py         # Synthetic data generation script
├── model.py            # CLV prediction model implementation
├── index.html          # Web interface
├── requirements.txt    # Project dependencies
├── data/               # Generated datasets
│   ├── customers.csv
│   ├── products.csv
│   ├── transactions.csv
│   └── customer_metrics.csv
└── models/            # Trained model artifacts
    └── clv_model_latest/
        ├── model.joblib
        ├── scaler.joblib
        ├── label_encoders.joblib
        ├── feature_names.joblib
        └── feature_importance.joblib
```

## Features

- **Data Generation**
  - Realistic synthetic customer data
  - Transaction history generation
  - Product holdings simulation
  - Configurable parameters for data size and date ranges

- **Machine Learning Model**
  - Gradient Boosting Regressor
  - Feature engineering pipeline
  - Model persistence and versioning
  - Confidence score calculation

- **Web API & Interface**
  - RESTful endpoints for predictions
  - Interactive web UI for data input
  - Real-time CLV predictions
  - Error handling and validation

## Installation

1. Clone the repository:
```bash
git clone https://github.com/Ismat-Samadov/clv_model.git
cd clv_model
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### 1. Generate Synthetic Data

Run the data generation script:
```bash
python data_gen.py
```
This will create synthetic banking data in the `data/` directory.

### 2. Train the Model

The model training is handled automatically when running the API for the first time. To manually train:
```python
from model import BankingCLVModel
model = BankingCLVModel()
model.train(customers_df, transactions_df, products_df, metrics_df)
```

### 3. Start the API Server

Run the FastAPI server:
```bash
uvicorn api:app --reload
```
The server will start at `http://localhost:8000`

### 4. Access the Web Interface

Open `http://localhost:8000` in your web browser to access the CLV prediction interface.

## API Endpoints

### POST /predict/clv
Predicts CLV for a given customer profile.

Example request body:
```json
{
  "customer_id": 1,
  "age": 35,
  "income": 75000,
  "credit_score": 720,
  "tenure_months": 24,
  "region": "North",
  "acquisition_channel": "Online",
  "products": [
    {
      "product_type": "Savings",
      "start_date": "2023-01-01T00:00:00",
      "balance": 5000,
      "status": "Active"
    }
  ],
  "transactions": [
    {
      "transaction_date": "2024-01-01T10:00:00",
      "transaction_type": "Deposit",
      "amount": 1000,
      "channel": "Online"
    }
  ]
}
```

### GET /health
Returns the health status of the API and model.

## Model Details

The CLV prediction model:
- Uses a Gradient Boosting Regressor
- Incorporates customer demographics, product holdings, and transaction patterns
- Provides confidence scores for predictions
- Includes feature importance analysis

### Key Features Used:
- Customer demographics (age, income, credit score)
- Account tenure
- Transaction patterns
- Product holdings
- Regional indicators
- Channel preferences

## Configuration

Key configuration options are available in the respective Python files:
- `data_gen.py`: Data generation parameters
- `model.py`: Model hyperparameters
- `api.py`: API settings and validation rules

## Logging

The system logs important events and errors to `api.log`. Configure logging levels in `api.py`.

## Development

### Adding New Features
1. Modify the data generation in `data_gen.py`
2. Update feature engineering in `model.py`
3. Add new endpoints in `api.py`
4. Update the web interface in `index.html`

### Testing
Run the development server with:
```bash
uvicorn api:app --reload --port 8000
```

## Security Considerations

- Input validation for all API endpoints
- Error handling for invalid data
- Confidence score calculation for predictions
- Rate limiting for API endpoints (TODO)

## Dependencies

Major dependencies include:
- FastAPI
- scikit-learn
- pandas
- numpy
- joblib
- uvicorn

See `requirements.txt` for complete list.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Commit changes
4. Push to the branch
5. Create a Pull Request


## Author

Ismat Samadov
