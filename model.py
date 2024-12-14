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
        
        # Perform cross-validation
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
        
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_pred_proba, multi_class='ovr'),
            'classification_report': classification_report(y_test, y_pred, output_dict=True)
        }
        
        return metrics

    def _plot_model_comparison(self, viz_path):
        """Plot model comparison results"""
        metrics = ['accuracy', 'roc_auc']
        scores = {model_type: [results['metrics'][m] for m in metrics] 
                 for model_type, results in self.evaluation_results.items()}
        
        plt.figure(figsize=(10, 6))
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (model_type, scores) in enumerate(scores.items()):
            plt.bar(x + i*width, scores, width, label=model_type.upper())
        
        plt.xlabel('Metrics')
        plt.ylabel('Score')
        plt.title('Model Performance Comparison')
        plt.xticks(x + width, metrics)
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(viz_path, 'model_comparison.png'))
        plt.close()

    def _plot_feature_importance(self, viz_path):
        """Plot feature importance for the best model"""
        if hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            indices = np.argsort(importance)[::-1][:20]  # Top 20 features
            
            plt.figure(figsize=(12, 8))
            plt.title(f'Top 20 Feature Importances ({self.best_model_type.upper()})')
            plt.bar(range(20), importance[indices])
            plt.xticks(range(20), [self.feature_names[i] for i in indices], rotation=45, ha='right')
            plt.tight_layout()
            plt.savefig(os.path.join(viz_path, 'feature_importance.png'))
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