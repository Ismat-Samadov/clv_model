import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, accuracy_score
import xgboost as xgb
import lightgbm as lgb
import optuna
from datetime import datetime
import joblib
import warnings
import json
import os
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class BankingModelBuilder:
    def __init__(self, data_dir="data", model_dir="trained_models"):
        self.data_dir = data_dir
        self.model_dir = model_dir
        self.models = {}
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.feature_names = None
        self.best_model = None
        self.best_model_type = None
        
        # Create model directory
        Path(self.model_dir).mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self.setup_logging()
    
    def setup_logging(self):
        """Setup logging for the model building process"""
        self.log_file = os.path.join(self.model_dir, "model_building_log.txt")
        self.log(f"Started model building process at {datetime.now()}")

    def log(self, message: str):
        """Log message to file"""
        with open(self.log_file, 'a') as f:
            f.write(f"{datetime.now()}: {message}\n")
        print(message)

    def evaluate_model(self, model, X_test: pd.DataFrame, y_test: pd.Series) -> Dict:
        """Comprehensive model evaluation"""
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)
        
        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred),
            'feature_importance': self.get_feature_importance(model, X_test.columns)
        }
        
        return metrics

    def get_feature_importance(self, model, feature_names) -> Dict:
        """Extract feature importance from the model"""
        importance_dict = {}
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
            for name, importance in zip(feature_names, importances):
                importance_dict[name] = float(importance)
        
        return importance_dict

    def train_models_with_evaluation(self, X: pd.DataFrame, y: pd.Series, n_trials: int = 50):
        """Train models with comprehensive evaluation"""
        self.log("Starting model training with evaluation...")
        
        # Store evaluation results
        self.evaluation_results = {}
        best_overall_score = 0
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        model_types = ['rf', 'xgb', 'lgb']
        
        for model_type in model_types:
            self.log(f"\nOptimizing {model_type.upper()} model...")
            
            # Optimize hyperparameters
            study = optuna.create_study(direction='maximize')
            study.optimize(
                lambda trial: self.objective(trial, X_train, y_train, model_type),
                n_trials=n_trials
            )
            
            # Train model with best parameters
            best_params = study.best_params
            self.log(f"Best parameters for {model_type}: {best_params}")
            
            if model_type == 'rf':
                model = RandomForestClassifier(**best_params, random_state=42)
            elif model_type == 'xgb':
                model = xgb.XGBClassifier(**best_params, random_state=42)
            else:  # lgb
                model = lgb.LGBMClassifier(**best_params, random_state=42)
            
            # Train and evaluate
            model.fit(X_train, y_train)
            self.models[model_type] = model
            
            # Evaluate
            metrics = self.evaluate_model(model, X_test, y_test)
            self.evaluation_results[model_type] = {
                'metrics': metrics,
                'parameters': best_params,
                'best_cv_score': study.best_value
            }
            
            self.log(f"\n{model_type.upper()} Model Performance:")
            self.log(f"ROC AUC Score: {metrics['roc_auc']:.4f}")
            self.log(f"Accuracy Score: {metrics['accuracy']:.4f}")
            self.log("\nClassification Report:")
            self.log(metrics['classification_report'])
            
            # Update best model
            if metrics['roc_auc'] > best_overall_score:
                best_overall_score = metrics['roc_auc']
                self.best_model = model
                self.best_model_type = model_type
        
        self.log(f"\nBest performing model: {self.best_model_type.upper()} with ROC AUC: {best_overall_score:.4f}")

    def save_deployment_files(self):
        """Save all necessary files for model deployment"""
        self.log("\nSaving deployment files...")
        
        # Save best model
        joblib.dump(self.best_model, os.path.join(self.model_dir, "best_model.joblib"))
        
        # Save preprocessors
        joblib.dump(self.scaler, os.path.join(self.model_dir, "scaler.joblib"))
        joblib.dump(self.label_encoders, os.path.join(self.model_dir, "label_encoders.joblib"))
        
        # Save feature names
        with open(os.path.join(self.model_dir, "feature_names.json"), 'w') as f:
            json.dump(list(self.feature_names), f)
        
        # Save evaluation results and metadata
        metadata = {
            'creation_date': datetime.now().isoformat(),
            'best_model': {
                'model_type': self.best_model_type,
                'metrics': self.evaluation_results[self.best_model_type],
                'parameters': self.best_model.get_params()
            },
            'model_comparisons': self.evaluation_results,
            'feature_names': list(self.feature_names),
            'target_classes': list(self.label_encoders.get('risk_category', []).classes_)
        }
        
        with open(os.path.join(self.model_dir, "model_metadata.json"), 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save feature importance plot
        self.plot_feature_importance()
        
        self.log("Saved all deployment files successfully!")

    def plot_feature_importance(self):
        """Plot feature importance for the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Feature Importance ({self.best_model_type.upper()})')
            plt.bar(range(len(importances)), importances[indices])
            plt.xticks(range(len(importances)), 
                      [self.feature_names[i] for i in indices], 
                      rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(self.model_dir, 'feature_importance.png'))
            plt.close()

def main():
    # Initialize model builder
    model_builder = BankingModelBuilder()
    
    # Load and prepare data
    model_builder.log("Loading and preparing data...")
    data = model_builder.load_and_merge_data()
    X, y = model_builder.prepare_features(data)
    model_builder.feature_names = X.columns
    
    # Train and evaluate models
    model_builder.train_models_with_evaluation(X, y, n_trials=50)
    
    # Save deployment files
    model_builder.save_deployment_files()

if __name__ == "__main__":
    main()