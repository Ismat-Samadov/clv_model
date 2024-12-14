import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from datetime import datetime
import joblib
import warnings
from typing import Dict, List, Tuple
import os

class BankingModelBuilder:
    def __init__(self, data_dir="data"):
        self.data_dir = data_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        
    def load_and_merge_data(self) -> pd.DataFrame:
        """Load and merge all banking datasets"""
        print("Loading and merging data...")
        
        # Load all datasets
        dfs = {
            'customers': pd.read_csv(f"{self.data_dir}/customers.csv"),
            'checking': pd.read_csv(f"{self.data_dir}/checking_accounts.csv"),
            'savings': pd.read_csv(f"{self.data_dir}/savings_accounts.csv"),
            'credit_cards': pd.read_csv(f"{self.data_dir}/credit_cards.csv"),
            'loans': pd.read_csv(f"{self.data_dir}/loans.csv"),
            'checking_txn': pd.read_csv(f"{self.data_dir}/checking_transactions.csv"),
            'savings_txn': pd.read_csv(f"{self.data_dir}/savings_transactions.csv"),
            'cc_txn': pd.read_csv(f"{self.data_dir}/credit_card_transactions.csv"),
            'loan_txn': pd.read_csv(f"{self.data_dir}/loan_transactions.csv"),
            'risk_metrics': pd.read_csv(f"{self.data_dir}/risk_metrics.csv")
        }
        
        # Calculate transaction aggregates per account
        checking_agg = self._aggregate_transactions(dfs['checking_txn'], 'checking')
        savings_agg = self._aggregate_transactions(dfs['savings_txn'], 'savings')
        cc_agg = self._aggregate_transactions(dfs['cc_txn'], 'credit_card')
        loan_agg = self._aggregate_transactions(dfs['loan_txn'], 'loan')
        
        # Merge account-level features
        accounts_merged = self._merge_account_features(
            dfs['checking'], dfs['savings'], dfs['credit_cards'], dfs['loans'],
            checking_agg, savings_agg, cc_agg, loan_agg
        )
        
        # Final merge with customer data and risk metrics
        final_df = dfs['customers'].merge(
            accounts_merged, on='customer_id', how='left'
        ).merge(
            dfs['risk_metrics'], on='customer_id', how='left'
        )
        
        print(f"Final dataset shape: {final_df.shape}")
        return final_df
    
    def _aggregate_transactions(self, df: pd.DataFrame, account_type: str) -> pd.DataFrame:
        """Aggregate transactions at account level"""
        if df.empty:
            return pd.DataFrame()
            
        aggs = df.groupby('account_id').agg({
            'amount': ['count', 'mean', 'std', 'sum', 'max', 'min'],
            'transaction_type': lambda x: x.nunique(),
            'status': lambda x: (x != 'COMPLETED').mean()
        }).round(2)
        
        # Flatten column names
        aggs.columns = [f"{account_type}_{col[0]}_{col[1]}" for col in aggs.columns]
        return aggs
    
    def _merge_account_features(self, checking, savings, credit_cards, loans,
                              checking_agg, savings_agg, cc_agg, loan_agg) -> pd.DataFrame:
        """Merge all account-level features"""
        account_features = []
        
        # Process checking accounts
        if not checking.empty:
            checking_features = checking.merge(checking_agg, on='account_id', how='left')
            checking_agg = checking_features.groupby('customer_id').agg({
                'current_balance': ['count', 'sum', 'mean'],
                'checking_amount_sum': 'sum',
                'checking_amount_mean': 'mean',
                'checking_amount_std': 'mean'
            }).round(2)
            checking_agg.columns = [f"checking_{col[0]}_{col[1]}" for col in checking_agg.columns]
            account_features.append(checking_agg)
            
        # Process savings accounts (similar to checking)
        if not savings.empty:
            savings_features = savings.merge(savings_agg, on='account_id', how='left')
            savings_agg = savings_features.groupby('customer_id').agg({
                'current_balance': ['count', 'sum', 'mean'],
                'savings_amount_sum': 'sum',
                'savings_amount_mean': 'mean',
                'savings_amount_std': 'mean'
            }).round(2)
            savings_agg.columns = [f"savings_{col[0]}_{col[1]}" for col in savings_agg.columns]
            account_features.append(savings_agg)
        
        # Process credit cards
        if not credit_cards.empty:
            cc_features = credit_cards.merge(cc_agg, on='account_id', how='left')
            cc_agg = cc_features.groupby('customer_id').agg({
                'credit_limit': ['count', 'sum', 'mean'],
                'utilization_rate': 'mean',
                'credit_card_amount_sum': 'sum',
                'credit_card_amount_mean': 'mean'
            }).round(2)
            cc_agg.columns = [f"cc_{col[0]}_{col[1]}" for col in cc_agg.columns]
            account_features.append(cc_agg)
        
        # Process loans
        if not loans.empty:
            loan_features = loans.merge(loan_agg, on='account_id', how='left')
            loan_agg = loan_features.groupby('customer_id').agg({
                'original_amount': ['count', 'sum', 'mean'],
                'current_balance': 'sum',
                'missed_payment_count': 'sum',
                'loan_amount_sum': 'sum'
            }).round(2)
            loan_agg.columns = [f"loan_{col[0]}_{col[1]}" for col in loan_agg.columns]
            account_features.append(loan_agg)
        
        # Combine all features
        return pd.concat([df for df in account_features if not df.empty], 
                       axis=1, join='outer').fillna(0)
    
    def prepare_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features for modeling"""
        print("Preparing features...")
        
        # Define target variable (risk_category)
        y = df['risk_category']
        
        # Drop unnecessary columns
        drop_cols = ['customer_id', 'ssn', 'email', 'phone_number', 'birth_date',
                    'onboarding_date', 'risk_category', 'calculation_date']
        X = df.drop(columns=drop_cols, errors='ignore')
        
        # Handle categorical variables
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        # Scale numerical features
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X, y
    
    def objective(self, trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, model_type: str):
        """Optuna objective function for hyperparameter optimization"""
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            model = RandomForestClassifier(**params, random_state=42)
            
        elif model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 10),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=42)
            
        elif model_type == 'lgb':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 200),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 1, 50),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = lgb.LGBMClassifier(**params, random_state=42)
        
        # Perform cross-validation
        scores = cross_val_score(model, X, y, cv=5, scoring='roc_auc')
        return scores.mean()
    
    def train_models(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
        """Train and optimize multiple models"""
        print("Training models...")
        
        model_types = ['rf', 'xgb', 'lgb']
        best_params = {}
        
        for model_type in model_types:
            print(f"\nOptimizing {model_type.upper()} model...")
            study = optuna.create_study(direction='maximize')
            study.optimize(lambda trial: self.objective(trial, X, y, model_type),
                         n_trials=n_trials)
            
            best_params[model_type] = study.best_params
            print(f"Best {model_type.upper()} parameters:", study.best_params)
            print(f"Best score: {study.best_value:.4f}")
            
            # Train final model with best parameters
            if model_type == 'rf':
                self.models[model_type] = RandomForestClassifier(
                    **best_params[model_type], random_state=42
                )
            elif model_type == 'xgb':
                self.models[model_type] = xgb.XGBClassifier(
                    **best_params[model_type], random_state=42
                )
            else:  # lgb
                self.models[model_type] = lgb.LGBMClassifier(
                    **best_params[model_type], random_state=42
                )
            
            # Train and evaluate final model
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )
            
            self.models[model_type].fit(X_train, y_train)
            y_pred = self.models[model_type].predict(X_test)
            
            print(f"\n{model_type.upper()} Model Performance:")
            print(classification_report(y_test, y_pred))
    
    def save_models(self, model_dir="models"):
        """Save trained models and preprocessors"""
        os.makedirs(model_dir, exist_ok=True)
        
        for model_type, model in self.models.items():
            joblib.dump(model, f"{model_dir}/{model_type}_model.joblib")
        
        joblib.dump(self.scaler, f"{model_dir}/scaler.joblib")
        joblib.dump(self.label_encoders, f"{model_dir}/label_encoders.joblib")
        
    def load_models(self, model_dir="models"):
        """Load trained models and preprocessors"""
        model_types = ['rf', 'xgb', 'lgb']
        
        for model_type in model_types:
            self.models[model_type] = joblib.load(f"{model_dir}/{model_type}_model.joblib")
        
        self.scaler = joblib.load(f"{model_dir}/scaler.joblib")
        self.label_encoders = joblib.load(f"{model_dir}/label_encoders.joblib")

if __name__ == "__main__":
    # Initialize model builder
    model_builder = BankingModelBuilder()
    
    # Load and prepare data
    data = model_builder.load_and_merge_data()
    X, y = model_builder.prepare_features(data)
    
    # Train and optimize models
    model_builder.train_models(X, y, n_trials=50)
    
    # Save models
    model_builder.save_models()