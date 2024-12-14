import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import optuna
import joblib
import json
import os
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime

class BankingBehaviorModel:
    def __init__(self, data_dir="data", model_dir="trained_models"):
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_model_type = None
        
        self.model_dir.mkdir(parents=True, exist_ok=True)

    def load_and_process_data(self):
        """Load and process all banking data with behavioral features"""
        print("Loading datasets...")
        
        # Load all datasets
        datasets = self._load_datasets()
        
        print("Creating behavioral features...")
        features = {}
        
        # Create different types of features
        account_features = self._create_account_features(datasets)
        transaction_features = self._create_transaction_features(datasets)
        merchant_features = self._create_merchant_features(datasets)
        product_features = self._create_product_features(datasets)
        
        # Combine all features
        features.update(account_features)
        features.update(transaction_features)
        features.update(merchant_features)
        features.update(product_features)
        
        # Create final dataset
        final_df = self._combine_features(
            datasets['customers'], 
            features, 
            datasets['risk_metrics']
        )
        
        print(f"Final dataset shape: {final_df.shape}")
        return final_df

    def _load_datasets(self):
        """Load all required datasets"""
        datasets = {}
        required_files = [
            'customers.csv', 'checking_accounts.csv', 'savings_accounts.csv',
            'credit_cards.csv', 'loans.csv', 'checking_transactions.csv',
            'savings_transactions.csv', 'credit_card_transactions.csv',
            'loan_transactions.csv', 'merchants.csv', 'products.csv',
            'branches.csv', 'risk_metrics.csv'
        ]
        
        for file in required_files:
            name = file.replace('.csv', '')
            file_path = self.data_dir / file
            try:
                datasets[name] = pd.read_csv(file_path)
                print(f"Loaded {file} with {len(datasets[name])} records")
            except FileNotFoundError:
                print(f"Warning: {file} not found in {self.data_dir}")
                datasets[name] = pd.DataFrame()
        
        return datasets

    def _create_account_features(self, datasets):
        """Create account-related behavioral features"""
        account_features = pd.DataFrame()
        
        # Process each account type
        account_types = {
            'checking': 'checking_accounts',
            'savings': 'savings_accounts',
            'credit': 'credit_cards',
            'loan': 'loans'
        }
        
        for acc_type, file_name in account_types.items():
            if file_name in datasets and not datasets[file_name].empty:
                df = datasets[file_name]
                
                # Basic account metrics
                metrics = df.groupby('customer_id').agg({
                    'account_id': 'count',
                    'current_balance': ['mean', 'sum', 'std']
                })
                
                # Flatten column names
                metrics.columns = [f'{acc_type}_{col[0]}_{col[1]}'.lower() 
                                 if isinstance(col, tuple) 
                                 else f'{acc_type}_{col}'.lower() 
                                 for col in metrics.columns]
                
                # Add type-specific features
                if acc_type == 'credit' and 'utilization_rate' in df.columns:
                    metrics[f'{acc_type}_utilization_mean'] = \
                        df.groupby('customer_id')['utilization_rate'].mean()
                
                if acc_type == 'loan' and 'original_amount' in df.columns:
                    metrics[f'{acc_type}_total_original'] = \
                        df.groupby('customer_id')['original_amount'].sum()
                
                # Update account features
                if account_features.empty:
                    account_features = metrics
                else:
                    account_features = account_features.join(metrics, how='outer')
        
        return account_features.fillna(0)

    def _create_transaction_features(self, datasets):
        """Create transaction-related behavioral features"""
        transaction_features = pd.DataFrame()
        
        # Process each transaction type
        transaction_types = {
            'checking': ('checking_transactions', 'checking_accounts'),
            'savings': ('savings_transactions', 'savings_accounts'),
            'credit': ('credit_card_transactions', 'credit_cards'),
            'loan': ('loan_transactions', 'loans')
        }
        
        for tx_type, (tx_file, acc_file) in transaction_types.items():
            if tx_file in datasets and acc_file in datasets and \
               not datasets[tx_file].empty and not datasets[acc_file].empty:
                
                # Merge transactions with accounts to get customer_id
                txn_df = datasets[tx_file].merge(
                    datasets[acc_file][['account_id', 'customer_id']],
                    on='account_id',
                    how='left'
                )
                
                # Calculate transaction metrics
                metrics = txn_df.groupby('customer_id').agg({
                    'amount': ['count', 'mean', 'sum', 'std'],
                    'transaction_id': 'nunique'
                })
                
                # Flatten column names
                metrics.columns = [f'{tx_type}_tx_{col[0]}_{col[1]}'.lower() 
                                 if isinstance(col, tuple) 
                                 else f'{tx_type}_tx_{col}'.lower() 
                                 for col in metrics.columns]
                
                # Update transaction features
                if transaction_features.empty:
                    transaction_features = metrics
                else:
                    transaction_features = transaction_features.join(metrics, how='outer')
        
        return transaction_features.fillna(0)

    def _create_merchant_features(self, datasets):
        """Create merchant interaction behavioral features"""
        merchant_features = pd.DataFrame()
        
        if 'merchants' in datasets and not datasets['merchants'].empty:
            # Combine credit card and checking transactions
            tx_data = []
            
            if 'credit_card_transactions' in datasets and not datasets['credit_card_transactions'].empty:
                cc_tx = datasets['credit_card_transactions'].merge(
                    datasets['credit_cards'][['account_id', 'customer_id']],
                    on='account_id',
                    how='left'
                )
                tx_data.append(cc_tx)
            
            if 'checking_transactions' in datasets and not datasets['checking_transactions'].empty:
                check_tx = datasets['checking_transactions'].merge(
                    datasets['checking_accounts'][['account_id', 'customer_id']],
                    on='account_id',
                    how='left'
                )
                tx_data.append(check_tx)
            
            if tx_data:
                # Combine all transactions
                all_tx = pd.concat(tx_data)
                
                # Merge with merchant data
                tx_merchants = all_tx.merge(
                    datasets['merchants'][['merchant_id', 'category']],
                    on='merchant_id',
                    how='left'
                )
                
                # Calculate merchant metrics
                merchant_features = tx_merchants.groupby('customer_id').agg({
                    'merchant_id': 'nunique',
                    'category': 'nunique',
                    'amount': ['sum', 'mean', 'std']
                })
                
                # Flatten column names
                merchant_features.columns = [f'merchant_{col[0]}_{col[1]}'.lower() 
                                          if isinstance(col, tuple) 
                                          else f'merchant_{col}'.lower() 
                                          for col in merchant_features.columns]
        
        return merchant_features.fillna(0)

    def _create_product_features(self, datasets):
        """Create product usage and relationship features"""
        product_features = pd.DataFrame()
        
        # Combine all account types
        account_dfs = []
        account_types = {
            'checking': 'checking_accounts',
            'savings': 'savings_accounts',
            'credit': 'credit_cards',
            'loan': 'loans'
        }
        
        for acc_type, file_name in account_types.items():
            if file_name in datasets and not datasets[file_name].empty:
                df = datasets[file_name][['customer_id', 'account_id']].copy()
                df['product_type'] = acc_type
                account_dfs.append(df)
        
        if account_dfs:
            all_accounts = pd.concat(account_dfs)
            
            # Calculate product metrics
            product_features = all_accounts.groupby('customer_id').agg({
                'product_type': 'nunique',
                'account_id': 'count'
            }).rename(columns={
                'product_type': 'product_diversity',
                'account_id': 'total_accounts'
            })
        
        return product_features.fillna(0)

    def _combine_features(self, customers_df, features_dict, risk_metrics_df):
        """Combine all features into final dataset"""
        final_df = customers_df.copy()
        
        # Add all feature groups
        for feature_df in features_dict.values():
            if not feature_df.empty:
                final_df = final_df.merge(feature_df, 
                                        on='customer_id', 
                                        how='left')
        
        # Add risk metrics
        if not risk_metrics_df.empty:
            final_df = final_df.merge(risk_metrics_df,
                                    on='customer_id',
                                    how='left')
        
        return final_df.fillna(0)

    
    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        # Define target variable
        y = df['risk_category']
        
        # Drop unnecessary columns
        drop_cols = ['customer_id', 'ssn', 'email', 'phone_number', 'birth_date',
                    'onboarding_date', 'risk_category', 'calculation_date',
                    'first_name', 'last_name', 'address_street', 'address_city']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        # Handle categorical variables
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X, y


def main():
    # Initialize model
    model = BankingBehaviorModel()
    
    # Process data
    print("Loading and processing data...")
    data = model.load_and_process_data()
    
    print("Preparing features...")
    X, y = model.prepare_features(data)
    model.feature_names = X.columns
    
    # Train and evaluate models
    print("Training and evaluating models...")
    model.train_and_evaluate_models(X, y, n_trials=50)
    
    # Save artifacts
    print("Saving model artifacts...")
    model.save_model_artifacts()
    
    # Print final results
    print("\nModel Training Results:")
    for model_type, results in model.evaluation_results.items():
        print(f"\n{model_type.upper()} Model:")
        print(f"ROC AUC: {results['metrics']['roc_auc']:.4f}")
        print(f"Accuracy: {results['metrics']['accuracy']:.4f}")
    
    print(f"\nBest Model: {model.best_model_type.upper()}")
    print("Model building completed successfully!")

if __name__ == "__main__":
    main()