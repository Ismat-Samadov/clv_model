import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import GradientBoostingRegressor
import joblib
import os
from datetime import datetime


class BankingCLVModel:
    def __init__(self, model_dir='models'):
        self.model_dir = model_dir
        self.model = None
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None  # Will store feature names in correct order
        
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)

    def prepare_features(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Feature engineering and preparation with fixed feature ordering
        """
        # Create base features DataFrame
        features_df = customers_df.copy()
        
        # Add metrics
        features_df = features_df.merge(metrics_df, on='customer_id')
        
        # Define all possible product types in fixed order
        all_product_types = [
            'savings', 'checking', 'credit card', 'mortgage', 
            'investment', 'personal loan'
        ]
        
        # Calculate product-related features
        product_counts = products_df.groupby('customer_id')['product_type'].value_counts().unstack().fillna(0)
        
        # Create product dummy columns in fixed order
        for product_type in all_product_types:
            col_name = f'has_product_{product_type.lower().replace(" ", "_")}'
            if product_type.upper() in product_counts.columns:
                features_df[col_name] = product_counts[product_type.upper()]
            else:
                features_df[col_name] = 0
        
        # Calculate transaction patterns
        if len(transactions_df) > 0:
            transaction_patterns = transactions_df.groupby('customer_id').agg({
                'amount': ['mean', 'std', 'min', 'max'],
                'transaction_type': 'count'
            }).reset_index()
            
            transaction_patterns.columns = [
                'customer_id', 'avg_transaction', 'std_transaction',
                'min_transaction', 'max_transaction', 'transaction_count'
            ]
        else:
            transaction_patterns = pd.DataFrame({
                'customer_id': [customers_df['customer_id'].iloc[0]],
                'avg_transaction': [0],
                'std_transaction': [0],
                'min_transaction': [0],
                'max_transaction': [0],
                'transaction_count': [0]
            })
        
        features_df = features_df.merge(transaction_patterns, on='customer_id', how='left')
        
        # Fill missing values
        features_df = features_df.fillna(0)
        
        # Encode categorical variables
        categorical_columns = ['region', 'acquisition_channel']
        for col in categorical_columns:
            if col in features_df.columns:
                if col not in self.label_encoders:
                    self.label_encoders[col] = LabelEncoder()
                features_df[col] = self.label_encoders[col].fit_transform(features_df[col])
        
        # Ensure all expected features are present
        if self.feature_names is not None:
            missing_cols = set(self.feature_names) - set(features_df.columns)
            for col in missing_cols:
                features_df[col] = 0
            
            # Reorder columns to match training order
            features_df = features_df[['customer_id'] + self.feature_names]
        
        return features_df

    def predict(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Make CLV predictions using the trained model
        """
        try:
            # Prepare features using the same preprocessing as during training
            features_df = self.prepare_features(customers_df, transactions_df, products_df, metrics_df)
            
            # Remove customer_id before scaling
            customer_ids = features_df['customer_id']
            X = features_df.drop('customer_id', axis=1)
            
            # Ensure columns match training data
            if self.feature_names:
                missing_cols = set(self.feature_names) - set(X.columns)
                for col in missing_cols:
                    X[col] = 0
                X = X[self.feature_names]
            
            # Scale features
            X_scaled = self.scaler.transform(X)
            
            # Make prediction using the trained model
            predictions = self.model.predict(X_scaled)
            
            return pd.DataFrame({
                'customer_id': customer_ids,
                'predicted_clv': predictions
            })
            
        except Exception as e:
            print(f"Error in prediction: {str(e)}")
            raise
        
    @classmethod
    def load_model(cls, model_path):
        """
        Load a saved model and its transformers
        """
        instance = cls()
        try:
            instance.model = joblib.load(os.path.join(model_path, 'model.joblib'))
            instance.scaler = joblib.load(os.path.join(model_path, 'scaler.joblib'))
            instance.label_encoders = joblib.load(os.path.join(model_path, 'label_encoders.joblib'))
            instance.feature_names = joblib.load(os.path.join(model_path, 'feature_names.joblib'))
        except Exception as e:
            print(f"Warning: Could not load model components: {str(e)}")
        return instance

    def save_model(self, version=None):
        """
        Save the trained model and associated transformers
        """
        if self.model is None:
            raise ValueError("No trained model to save!")
            
        if version is None:
            version = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        model_path = os.path.join(self.model_dir, f'clv_model_{version}')
        os.makedirs(model_path, exist_ok=True)
        
        joblib.dump(self.model, os.path.join(model_path, 'model.joblib'))
        joblib.dump(self.scaler, os.path.join(model_path, 'scaler.joblib'))
        joblib.dump(self.label_encoders, os.path.join(model_path, 'label_encoders.joblib'))
        joblib.dump(self.feature_names, os.path.join(model_path, 'feature_names.joblib'))
        
        return model_path
    
    def calculate_current_clv(self, transactions_df, customers_df):
        """
        Calculate current CLV based on historical transactions
        """
        # Calculate average monthly value and multiply by expected customer lifetime
        monthly_value = transactions_df.groupby('customer_id')['amount'].sum() / \
                       customers_df['tenure_months']
        
        # Assume a 5-year future lifetime for this example
        future_months = 60
        discount_rate = 0.01  # Monthly discount rate
        
        # Calculate NPV of future cash flows
        clv = monthly_value * sum(1 / (1 + discount_rate)**i for i in range(future_months))
        
        return clv.reset_index(name='current_clv')
    
    def train(self, customers_df, transactions_df, products_df, metrics_df):
        """
        Train the CLV prediction model
        """
        # Prepare features
        features_df = self.prepare_features(customers_df, transactions_df, products_df, metrics_df)
        
        # Calculate current CLV (target variable)
        clv_df = self.calculate_current_clv(transactions_df, customers_df)
        features_df = features_df.merge(clv_df, on='customer_id')
        
        # Separate features and target
        X = features_df.drop(['customer_id', 'current_clv'], axis=1)
        y = features_df['current_clv']
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)
        
        # Train model
        self.model = GradientBoostingRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Calculate and print metrics
        train_score = self.model.score(X_train, y_train)
        test_score = self.model.score(X_test, y_test)
        
        print(f"Model R² score on training data: {train_score:.4f}")
        print(f"Model R² score on test data: {test_score:.4f}")
        
        # Save feature names for later use
        self.feature_names = X.columns.tolist()
        
        return train_score, test_score


# Example usage
if __name__ == "__main__":
    # Load the generated data
    data_dir = 'data'
    customers_df = pd.read_csv(f"{data_dir}/customers.csv")
    transactions_df = pd.read_csv(f"{data_dir}/transactions.csv")
    products_df = pd.read_csv(f"{data_dir}/products.csv")
    metrics_df = pd.read_csv(f"{data_dir}/customer_metrics.csv")
    
    # Convert transaction_date to datetime
    transactions_df['transaction_date'] = pd.to_datetime(transactions_df['transaction_date'])
    products_df['start_date'] = pd.to_datetime(products_df['start_date'])
    
    # Initialize and train the model
    clv_model = BankingCLVModel()
    train_score, test_score = clv_model.train(customers_df, transactions_df, products_df, metrics_df)
    
    # Make predictions
    predictions = clv_model.predict(customers_df, transactions_df, products_df, metrics_df)
    print("\nSample predictions:")
    print(predictions.head())
    
    # Save the model
    model_path = clv_model.save_model()
    
    # Example of loading and using the saved model
    loaded_model = BankingCLVModel.load_model(model_path)
    new_predictions = loaded_model.predict(customers_df, transactions_df, products_df, metrics_df)
    
    print("\nVerifying predictions with loaded model:")
    print(new_predictions.head())