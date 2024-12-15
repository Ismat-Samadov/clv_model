import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import logging
import os
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class BankingCLVModel:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.feature_importance = None
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def prepare_features(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Feature engineering with proper type handling
        """
        logger.info("Starting feature preparation...")
        features_df = customers_df.copy()
        
        # Define all possible categorical values
        ALL_CHANNELS = ['Online', 'Mobile', 'Branch', 'ATM']
        ALL_PRODUCTS = ['Savings', 'Checking', 'Credit Card', 'Mortgage', 'Investment', 'Personal Loan']
        ALL_TRANSACTION_TYPES = ['Deposit', 'Withdrawal', 'Transfer', 'Bill Payment', 'Fee']
        
        # Initialize all binary columns with proper integer dtype
        for channel in ALL_CHANNELS:
            features_df[f'channel_{channel}'] = pd.Series(0, index=features_df.index, dtype='int64')
        for product in ALL_PRODUCTS:
            features_df[f'has_product_{product}'] = pd.Series(0, index=features_df.index, dtype='int64')
        for tx_type in ALL_TRANSACTION_TYPES:
            features_df[f'prop_{tx_type.lower()}_transactions'] = pd.Series(0, index=features_df.index, dtype='float64')
        
        # Basic transaction metrics
        if len(transactions_df) > 0:
            # Transaction amounts
            tx_stats = transactions_df.groupby('customer_id').agg({
                'amount': ['count', 'sum', 'mean', 'std', 'min', 'max']
            }).reset_index()
            tx_stats.columns = ['customer_id', 'transaction_count', 'total_amount',
                            'avg_transaction', 'std_transaction', 'min_transaction',
                            'max_transaction']
            features_df = features_df.merge(tx_stats, on='customer_id', how='left')
            
            # Channel proportions
            channel_counts = pd.crosstab(
                transactions_df['customer_id'],
                transactions_df['channel'],
                normalize='index'
            )
            for channel in ALL_CHANNELS:
                if channel in channel_counts.columns:
                    channel_col = f'channel_{channel}'
                    features_df[channel_col] = features_df['customer_id'].map(
                        channel_counts[channel].astype('int64')
                    ).fillna(0)
            
            # Transaction type proportions
            type_counts = pd.crosstab(
                transactions_df['customer_id'],
                transactions_df['transaction_type'],
                normalize='index'
            )
            for tx_type in ALL_TRANSACTION_TYPES:
                if tx_type in type_counts.columns:
                    type_col = f'prop_{tx_type.lower()}_transactions'
                    features_df[type_col] = features_df['customer_id'].map(
                        type_counts[tx_type]
                    ).fillna(0)
        
        # Product metrics
        if len(products_df) > 0:
            # Product type indicators
            product_dummies = pd.get_dummies(products_df['product_type'], prefix='has_product')
            product_by_customer = product_dummies.groupby(products_df['customer_id']).max()
            for product in ALL_PRODUCTS:
                col = f'has_product_{product}'
                if col in product_by_customer.columns:
                    features_df[col] = features_df['customer_id'].map(
                        product_by_customer[col].astype('int64')
                    ).fillna(0)
            
            # Aggregate product metrics
            product_stats = products_df.groupby('customer_id').agg({
                'balance': ['sum', 'mean', 'max'],
                'product_type': 'count'
            }).reset_index()
            product_stats.columns = ['customer_id', 'total_balance', 'avg_balance',
                                'max_balance', 'product_count']
            features_df = features_df.merge(product_stats, on='customer_id', how='left')
        
        # Add derived features
        features_df['transaction_frequency'] = (
            features_df['transaction_count'] / features_df['tenure_months']
        )
        features_df['avg_monthly_value'] = (
            features_df['total_amount'] / features_df['tenure_months']
        )
        features_df['balance_to_income_ratio'] = (
            features_df['total_balance'] / features_df['income']
        )
        
        # Fill missing values with explicit dtype preservation
        numeric_cols = features_df.select_dtypes(include=['float64', 'int64']).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        # Encode categorical variables
        categorical_columns = ['region', 'acquisition_channel']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                    self.label_encoders[col].fit(features_df[col])
                features_df[col] = self.label_encoders[col].transform(features_df[col])
        
        # Ensure consistent feature order
        if self.feature_names is not None:
            # Add any missing columns
            for col in self.feature_names:
                if col not in features_df.columns:
                    features_df[col] = pd.Series(0, index=features_df.index, dtype='float64')
            # Reorder columns to match training
            return features_df[['customer_id'] + self.feature_names]
        else:
            # During training, save feature names
            self.feature_names = [col for col in features_df.columns if col != 'customer_id']
            return features_df
    
    def calculate_current_clv(self, transactions_df, customers_df):
        """
        Enhanced CLV calculation with improved business logic
        """
        logger.info("Calculating current CLV...")
        
        # Group transactions by customer
        customer_txns = transactions_df.groupby('customer_id')
        
        # Calculate monthly metrics
        monthly_stats = customer_txns.agg({
            'amount': ['sum', 'count', 'mean', 'std']
        }).reset_index()
        
        monthly_stats.columns = [
            'customer_id', 'total_amount', 'transaction_count',
            'avg_amount', 'std_amount'
        ]
        
        # Add customer tenure
        tenure_df = customers_df[['customer_id', 'tenure_months', 'income']]
        monthly_stats = monthly_stats.merge(tenure_df, on='customer_id')
        
        # Calculate monthly value
        monthly_stats['monthly_value'] = monthly_stats['total_amount'] / \
                                       monthly_stats['tenure_months'].clip(lower=1)
        
        # Project future value (5 years)
        future_months = 60
        discount_rate = 0.01  # Monthly discount rate
        
        # Calculate NPV with risk adjustment
        def calculate_npv(row):
            # Adjust discount rate based on transaction volatility
            volatility_factor = (row['std_amount'] / row['avg_amount']) if row['avg_amount'] > 0 else 1
            adjusted_rate = discount_rate * (1 + volatility_factor)
            
            # Calculate NPV
            npv_multiplier = sum(1 / (1 + adjusted_rate)**i for i in range(future_months))
            base_clv = row['monthly_value'] * npv_multiplier
            
            # Apply income-based bounds
            max_clv = row['income'] * 5  # Maximum 5 years of income
            return min(max_clv, max(0, base_clv))
        
        monthly_stats['current_clv'] = monthly_stats.apply(calculate_npv, axis=1)
        
        return monthly_stats[['customer_id', 'current_clv']]

    def train(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Enhanced model training with log transformation and bounds
        """
        try:
            logger.info("Starting model training...")
            
            # Calculate target variable
            clv_df = self.calculate_current_clv(transactions_df, customers_df)
            
            # Log transform target (after adding small constant to handle zeros)
            y_original = clv_df['current_clv']
            y_transformed = np.log1p(y_original)  # log1p(x) = log(1 + x)
            clv_df['current_clv'] = y_transformed
            
            logger.info(f"Original CLV range: {y_original.min():.2f} to {y_original.max():.2f}")
            logger.info(f"Transformed CLV range: {y_transformed.min():.2f} to {y_transformed.max():.2f}")
            
            # Prepare features
            features_df = self.prepare_features(customers_df, transactions_df, products_df, metrics_df)
            training_data = features_df.merge(clv_df, on='customer_id')
            
            # Split features and target
            X = training_data.drop(['customer_id', 'current_clv'], axis=1)
            y = training_data['current_clv']
            
            # Stratify by CLV quartiles
            y_quartiles = pd.qcut(y, q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y_quartiles
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Initialize model
            self.model = GradientBoostingRegressor(
                n_estimators=500,
                learning_rate=0.01,
                max_depth=3,
                min_samples_split=10,
                min_samples_leaf=5,
                subsample=0.7,
                max_features='sqrt',
                validation_fraction=0.1,
                n_iter_no_change=25,
                random_state=42
            )
            
            # Train model
            self.model.fit(X_train_scaled, y_train)
            
            # Calculate scores on transformed data
            train_score = self.model.score(X_train_scaled, y_train)
            test_score = self.model.score(X_test_scaled, y_test)
            
            logger.info(f"Training R² score (transformed): {train_score:.4f}")
            logger.info(f"Test R² score (transformed): {test_score:.4f}")
            
            # Feature importance
            feature_importance = dict(zip(X.columns, self.model.feature_importances_))
            self.feature_importance = dict(sorted(
                feature_importance.items(),
                key=lambda x: x[1],
                reverse=True
            ))
            
            logger.info("Top 5 important features:")
            for feature, importance in list(self.feature_importance.items())[:5]:
                logger.info(f"{feature}: {importance:.4f}")
            
            # Validate predictions with inverse transform
            train_preds = np.expm1(self.model.predict(X_train_scaled))  # inverse of log1p
            test_preds = np.expm1(self.model.predict(X_test_scaled))
            
            logger.info(f"Training predictions range: {train_preds.min():.2f} to {train_preds.max():.2f}")
            logger.info(f"Test predictions range: {test_preds.min():.2f} to {test_preds.max():.2f}")
            
            return train_score, test_score
            
        except Exception as e:
            logger.error(f"Error in training: {str(e)}", exc_info=True)
            raise
    
    def predict(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Final version of prediction with guaranteed positive CLV values
        """
        try:
            logger.info("Starting prediction process...")
            
            # Prepare features
            features_df = self.prepare_features(customers_df, transactions_df, products_df, metrics_df)
            
            # Extract customer IDs and features
            customer_ids = features_df['customer_id']
            X = features_df.drop('customer_id', axis=1)
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction and inverse transform
            log_predictions = self.model.predict(X_scaled)
            raw_predictions = np.expm1(log_predictions)
            
            # Apply bounds and business rules
            min_monthly_value = 0  # Minimum CLV
            max_multiplier = 5     # Maximum years of income to consider
            
            predictions = np.clip(raw_predictions, min_monthly_value, None)  # Ensure no negatives
            max_clv = customers_df['income'] * max_multiplier
            predictions = np.minimum(predictions, max_clv)  # Apply income-based cap
            
            # Calculate confidence based on prediction relative to bounds
            relative_position = predictions / max_clv
            confidence_base = self.calculate_confidence_scores(
                transactions_df, products_df, customers_df
            )
            
            # Adjust confidence based on prediction reasonableness
            confidence_scores = confidence_base * (1 - 0.2 * relative_position)  # Lower confidence for very high predictions
            confidence_scores = np.clip(confidence_scores, 0.1, 0.95)  # Keep confidence in reasonable range
            
            # Prepare results with metadata
            results = pd.DataFrame({
                'customer_id': customer_ids,
                'predicted_clv': predictions.round(2),
                'confidence_score': confidence_scores.round(3),
                'max_reasonable_clv': max_clv.round(2),
                'prediction_pct_of_max': (relative_position * 100).round(1)
            })
            
            # Log summary statistics
            logger.info(f"Predictions generated for {len(results)} customers")
            logger.info(f"Prediction range: {predictions.min():.2f} to {predictions.max():.2f}")
            logger.info(f"Average prediction: {predictions.mean():.2f}")
            logger.info(f"Median prediction: {np.median(predictions):.2f}")
            logger.info(f"Average confidence score: {confidence_scores.mean():.3f}")
            
            # Additional validation logging
            high_predictions = predictions[predictions > max_clv * 0.8]
            if len(high_predictions) > 0:
                logger.warning(f"Found {len(high_predictions)} predictions > 80% of maximum reasonable CLV")
                
            return results
            
        except Exception as e:
            logger.error(f"Error in prediction: {str(e)}", exc_info=True)
            raise

    def calculate_confidence_scores(self, transactions_df, products_df, customers_df):
        """
        Enhanced confidence score calculation
        """
        base_score = 0.7  # Base confidence
        
        # Transaction history score (up to 0.15)
        tx_count = len(transactions_df)
        tx_score = min(0.15, tx_count * 0.001)
        
        # Product diversity score (up to 0.1)
        unique_products = len(products_df['product_type'].unique())
        product_score = min(0.1, unique_products * 0.02)
        
        # Tenure score (up to 0.05)
        max_tenure = 120  # 10 years
        tenure_months = customers_df['tenure_months'].iloc[0]
        tenure_score = min(0.05, (tenure_months / max_tenure) * 0.05)
        
        # Calculate final score
        total_score = base_score + tx_score + product_score + tenure_score
        
        # Cap at 0.95 for some uncertainty
        return min(total_score, 0.95)
    
    @classmethod
    def load_model(cls, model_path):
        """Load a saved model and its transformers"""
        instance = cls()
        try:
            instance.model = joblib.load(os.path.join(model_path, 'model.joblib'))
            instance.scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
            instance.label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.joblib'))
            instance.feature_names = joblib.load(os.path.join(model_path, 'feature_names.joblib'))
            instance.feature_importance = joblib.load(os.path.join(model_path, 'feature_importance.joblib'))
            logger.info(f"Model successfully loaded from {model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {str(e)}")
            raise
        return instance

    def save_model(self, version=None):
        """Save the trained model and associated transformers"""
        if self.model is None:
            raise ValueError("No trained model to save!")
        
        version = version or datetime.now().strftime('%Y%m%d_%H%M%S')
        model_path = os.path.join(self.model_dir, f'clv_model_{version}')
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_path, 'model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(model_path, 'label_encoders.joblib'))
        joblib.dump(self.feature_names, os.path.join(model_path, 'feature_names.joblib'))
        joblib.dump(self.feature_importance, os.path.join(model_path, 'feature_importance.joblib'))
        
        logger.info(f"Model saved to {model_path}")
        return model_path

# Initialize and train
data_dir = 'data'
customers_df = pd.read_csv(f"{data_dir}/customers.csv")
transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
products_df = pd.read_csv(f"{data_dir}/products.csv")
metrics_df = pd.read_csv(f"{data_dir}/customer_metrics.csv")

# Convert dates
transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
products_df['start_date'] = pd.to_datetime(products_df['start_date'])

# Train model
model = BankingCLVModel()
train_score, test_score = model.train(customers_df, transactions_df, products_df, metrics_df)

# Save the model
model_path = model.save_model()

# Make predictions
predictions = model.predict(customers_df, transactions_df, products_df, metrics_df)