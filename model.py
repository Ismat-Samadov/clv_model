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
        """Initialize the Banking Behavior Model"""
        self.data_dir = Path(data_dir)
        self.model_dir = Path(model_dir)
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_model_type = None
        self.evaluation_results = {}
        
        self.model_dir.mkdir(parents=True, exist_ok=True)
        (self.model_dir / 'models').mkdir(exist_ok=True)
        (self.model_dir / 'visualizations').mkdir(exist_ok=True)

    def load_and_process_data(self):
        """Load and process all banking data with behavioral features"""
        print("Loading datasets...")
        datasets = self._load_datasets()
        
        print("Creating behavioral features...")
        features = {
            'account': self._create_account_features(datasets),
            'transaction': self._create_transaction_features(datasets),
            'merchant': self._create_merchant_features(datasets),
            'product': self._create_product_features(datasets)
        }
        
        final_df = self._combine_features(
            datasets['customers'], 
            features, 
            datasets['risk_metrics']
        )
        
        print(f"Final dataset shape: {final_df.shape}")
        return final_df

    def _load_datasets(self):
        """Load all required datasets"""
        required_files = [
            'customers.csv', 'checking_accounts.csv', 'savings_accounts.csv',
            'credit_cards.csv', 'loans.csv', 'checking_transactions.csv',
            'savings_transactions.csv', 'credit_card_transactions.csv',
            'loan_transactions.csv', 'merchants.csv', 'products.csv',
            'branches.csv', 'risk_metrics.csv'
        ]
        
        datasets = {}
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
        account_types = {
            'checking': 'checking_accounts',
            'savings': 'savings_accounts',
            'credit': 'credit_cards',
            'loan': 'loans'
        }
        
        for acc_type, file_name in account_types.items():
            if file_name in datasets and not datasets[file_name].empty:
                df = datasets[file_name]
                
                metrics = df.groupby('customer_id').agg({
                    'account_id': 'count',
                    'current_balance': ['mean', 'sum', 'std']
                })
                
                metrics.columns = [f'{acc_type}_{col[0]}_{col[1]}'.lower() 
                                 if isinstance(col, tuple) 
                                 else f'{acc_type}_{col}'.lower() 
                                 for col in metrics.columns]
                
                if acc_type == 'credit' and 'utilization_rate' in df.columns:
                    metrics[f'{acc_type}_utilization_mean'] = \
                        df.groupby('customer_id')['utilization_rate'].mean()
                
                if acc_type == 'loan' and 'original_amount' in df.columns:
                    metrics[f'{acc_type}_total_original'] = \
                        df.groupby('customer_id')['original_amount'].sum()
                
                if account_features.empty:
                    account_features = metrics
                else:
                    account_features = account_features.join(metrics, how='outer')
        
        return account_features.fillna(0)

    def _create_transaction_features(self, datasets):
        """Create transaction-related behavioral features"""
        transaction_features = pd.DataFrame()
        transaction_types = {
            'checking': ('checking_transactions', 'checking_accounts'),
            'savings': ('savings_transactions', 'savings_accounts'),
            'credit': ('credit_card_transactions', 'credit_cards'),
            'loan': ('loan_transactions', 'loans')
        }
        
        for tx_type, (tx_file, acc_file) in transaction_types.items():
            if tx_file in datasets and acc_file in datasets and \
               not datasets[tx_file].empty and not datasets[acc_file].empty:
                
                txn_df = datasets[tx_file].merge(
                    datasets[acc_file][['account_id', 'customer_id']],
                    on='account_id',
                    how='left'
                )
                
                metrics = txn_df.groupby('customer_id').agg({
                    'amount': ['count', 'mean', 'sum', 'std'],
                    'transaction_id': 'nunique'
                })
                
                metrics.columns = [f'{tx_type}_tx_{col[0]}_{col[1]}'.lower() 
                                 if isinstance(col, tuple) 
                                 else f'{tx_type}_tx_{col}'.lower() 
                                 for col in metrics.columns]
                
                if transaction_features.empty:
                    transaction_features = metrics
                else:
                    transaction_features = transaction_features.join(metrics, how='outer')
        
        return transaction_features.fillna(0)

    def _create_merchant_features(self, datasets):
        """Create merchant interaction behavioral features"""
        merchant_features = pd.DataFrame()
        
        if 'merchants' in datasets and not datasets['merchants'].empty:
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
                all_tx = pd.concat(tx_data)
                tx_merchants = all_tx.merge(
                    datasets['merchants'][['merchant_id', 'category']],
                    on='merchant_id',
                    how='left'
                )
                
                merchant_features = tx_merchants.groupby('customer_id').agg({
                    'merchant_id': 'nunique',
                    'category': 'nunique',
                    'amount': ['sum', 'mean', 'std']
                })
                
                merchant_features.columns = [f'merchant_{col[0]}_{col[1]}'.lower() 
                                          if isinstance(col, tuple) 
                                          else f'merchant_{col}'.lower() 
                                          for col in merchant_features.columns]
        
        return merchant_features.fillna(0)

    def _create_product_features(self, datasets):
        """Create product usage and relationship features"""
        product_features = pd.DataFrame()
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
        
        for feature_df in features_dict.values():
            if not feature_df.empty:
                final_df = final_df.merge(feature_df, 
                                        on='customer_id', 
                                        how='left')
        
        if not risk_metrics_df.empty:
            final_df = final_df.merge(risk_metrics_df,
                                    on='customer_id',
                                    how='left')
        
        return final_df.fillna(0)

    def prepare_features(self, df):
        """Prepare features for modeling"""
        print("Preparing features for modeling...")
        y = df['risk_category']
        
        drop_cols = ['customer_id', 'ssn', 'email', 'phone_number', 'birth_date',
                    'onboarding_date', 'risk_category', 'calculation_date',
                    'first_name', 'last_name', 'address_street', 'address_city']
        X = df.drop(columns=[col for col in drop_cols if col in df.columns])
        
        cat_columns = X.select_dtypes(include=['object']).columns
        for col in cat_columns:
            le = LabelEncoder()
            X[col] = le.fit_transform(X[col].astype(str))
            self.label_encoders[col] = le
        
        X = pd.DataFrame(
            self.scaler.fit_transform(X),
            columns=X.columns
        )
        
        return X, y

    def train_and_evaluate_models(self, X, y, n_trials=50):
        """Train and evaluate models with hyperparameter optimization"""
        print("Training and evaluating models...")
        best_overall_score = 0
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        for model_type in ['rf', 'xgb', 'lgb']:
            print(f"\nOptimizing {model_type.upper()} model...")
            
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self._objective(trial, X_train, y_train, model_type),
                n_trials=n_trials
            )
            
            model = self._create_and_train_model(model_type, study.best_params, X_train, y_train)
            self.models[model_type] = model
            
            metrics = self._evaluate_model(model, X_test, y_test)
            self.evaluation_results[model_type] = {
                'metrics': metrics,
                'parameters': study.best_params,
                'best_cv_score': study.best_value
            }
            
            if metrics['roc_auc'] > best_overall_score:
                best_overall_score = metrics['roc_auc']
                self.best_model = model
                self.best_model_type = model_type

    def _objective(self, trial, X_train, y_train, model_type):
        """Optimization objective for Optuna"""
        if model_type == 'rf':
            params = {
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 10),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 5)
            }
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        
        elif model_type == 'xgb':
            params = {
                'max_depth': trial.suggest_int('max_depth', 3, 15),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        
        elif model_type == 'lgb':
            params = {
                'num_leaves': trial.suggest_int('num_leaves', 20, 100),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
                'n_estimators': trial.suggest_int('n_estimators', 100, 500),
                'min_child_samples': trial.suggest_int('min_child_samples', 5, 30),
                'subsample': trial.suggest_float('subsample', 0.6, 1.0),
                'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0)
            }
            model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        
        cv_scores = cross_val_score(
            model, X_train, y_train, 
            cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
            scoring='roc_auc'
        )
        
        return cv_scores.mean()

    def _create_and_train_model(self, model_type, params, X_train, y_train):
        """Create and train a model with given parameters"""
        if model_type == 'rf':
            model = RandomForestClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type == 'xgb':
            model = xgb.XGBClassifier(**params, random_state=42, n_jobs=-1)
        elif model_type == 'lgb':
            model = lgb.LGBMClassifier(**params, random_state=42, n_jobs=-1)
        
        model.fit(X_train, y_train)
        return model

    def _evaluate_model(self, model, X_test, y_test):
        """Evaluate model performance"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        try:
            roc_auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr')
        except ValueError:
            # Fallback to binary classification if only two classes
            roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc,
            'classification_report': classification_report(y_test, y_pred, output_dict=True),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist()
        }
        
        return metrics

    def save_model_artifacts(self):
        """Save model artifacts and visualizations"""
        if not self.best_model:
            print("No model to save. Please train models first.")
            return
            
        # Create directories if they don't exist
        model_path = self.model_dir / 'models'
        viz_path = self.model_dir / 'visualizations'
        model_path.mkdir(exist_ok=True)
        viz_path.mkdir(exist_ok=True)
        
        # Save best model and preprocessors
        joblib.dump(self.best_model, model_path / 'best_model.joblib')
        joblib.dump(self.scaler, model_path / 'scaler.joblib')
        joblib.dump(self.label_encoders, model_path / 'label_encoders.joblib')
        
        # Save feature names
        with open(model_path / 'feature_names.json', 'w') as f:
            json.dump(list(self.feature_names), f)
        
        # Save model evaluation results
        with open(model_path / 'evaluation_results.json', 'w') as f:
            # Convert numpy types to native Python types for JSON serialization
            results = {}
            for model_type, result in self.evaluation_results.items():
                results[model_type] = {
                    'metrics': {
                        k: float(v) if isinstance(v, (np.float32, np.float64)) 
                        else v for k, v in result['metrics'].items()
                        if k != 'confusion_matrix'
                    },
                    'parameters': {
                        k: int(v) if isinstance(v, np.integer) 
                        else float(v) if isinstance(v, np.floating)
                        else v for k, v in result['parameters'].items()
                    },
                    'best_cv_score': float(result['best_cv_score'])
                }
            json.dump(results, f, indent=2)
        
        # Generate and save visualizations
        self._plot_model_comparison(viz_path)
        self._plot_feature_importance(viz_path)
        self._plot_confusion_matrices(viz_path)
        
        print(f"Model artifacts saved to {self.model_dir}")

    def _plot_model_comparison(self, viz_path):
        """Plot model comparison results"""
        metrics = ['accuracy', 'roc_auc']
        scores = {model_type: [results['metrics'][m] for m in metrics] 
                 for model_type, results in self.evaluation_results.items()}
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (model_type, model_scores) in enumerate(scores.items()):
            plt.bar(x + i*width, model_scores, width, label=model_type.upper())
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(viz_path / 'model_comparison.png')
        plt.close()

    def _plot_feature_importance(self, viz_path):
        """Plot feature importance for the best model"""
        if not hasattr(self.best_model, 'feature_importances_'):
            return
            
        importance = self.best_model.feature_importances_
        indices = np.argsort(importance)[::-1][:20]  # Top 20 features
        
        plt.figure(figsize=(12, 8))
        plt.title(f'Top 20 Feature Importances ({self.best_model_type.upper()})')
        plt.bar(range(20), importance[indices])
        plt.xticks(range(20), [self.feature_names[i] for i in indices], 
                  rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(viz_path / 'feature_importance.png')
        plt.close()

    def _plot_confusion_matrices(self, viz_path):
        """Plot confusion matrices for all models"""
        for model_type, results in self.evaluation_results.items():
            cm = np.array(results['metrics']['confusion_matrix'])
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title(f'Confusion Matrix - {model_type.upper()}')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            plt.tight_layout()
            plt.savefig(viz_path / f'confusion_matrix_{model_type}.png')
            plt.close()

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