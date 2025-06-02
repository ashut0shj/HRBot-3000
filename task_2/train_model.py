import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import cross_val_score, StratifiedKFold
from imblearn.over_sampling import SMOTE
import joblib
import json
import os

def train_models():
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    print(f"Original training distribution: {y_train.value_counts().to_dict()}")
    print(f"Test distribution: {y_test.value_counts().to_dict()}")
    
    if len(y_train.unique()) < 2:
        print("ERROR: Only one class in training data")
        return None
    
    smote = SMOTE(random_state=42, k_neighbors=min(5, y_train.sum()-1))
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    print(f"After SMOTE: {pd.Series(y_train_balanced).value_counts().to_dict()}")
    
    models = {
        'RandomForest': RandomForestClassifier(
            n_estimators=500,
            max_depth=15,
            min_samples_split=10,
            min_samples_leaf=5,
            max_features='sqrt',
            random_state=42,
            class_weight='balanced'
        ),
        'GradientBoosting': GradientBoostingClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            random_state=42
        ),
        'LogisticRegression': LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=3000,
            C=0.1,
            penalty='l2'
        )
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    results = {}
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    for name, model in models.items():
        print(f"\nTraining {name}...")
        
        cv_scores = cross_val_score(model, X_train_balanced, y_train_balanced, 
                                   cv=cv, scoring='roc_auc')
        
        model.fit(X_train_balanced, y_train_balanced)
        
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        results[name] = {
            'cv_auc_mean': cv_scores.mean(),
            'cv_auc_std': cv_scores.std(),
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc
        }
        
        print(f"CV AUC: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        print(f"Test AUC: {auc:.3f}")
        print(f"Test F1: {f1:.3f}")
        print(f"Precision: {precision:.3f}, Recall: {recall:.3f}")
        
        if auc > best_score:
            best_score = auc
            best_model = model
            best_name = name
    
    print(f"\nBest model: {best_name} with AUC: {best_score:.3f}")
    
    joblib.dump(best_model, 'attrition_model.pkl')
    
    os.makedirs('model_info', exist_ok=True)
    
    y_pred_final = best_model.predict(X_test)
    y_proba_final = best_model.predict_proba(X_test)[:, 1]
    
    print(f"\nFinal Model Performance:")
    print(classification_report(y_test, y_pred_final))
    
    high_risk = (y_proba_final > 0.7).sum()
    medium_risk = ((y_proba_final > 0.4) & (y_proba_final <= 0.7)).sum()
    low_risk = (y_proba_final <= 0.4).sum()
    
    print(f"\nRisk Distribution:")
    print(f"High Risk (>0.7): {high_risk}")
    print(f"Medium Risk (0.4-0.7): {medium_risk}")
    print(f"Low Risk (<0.4): {low_risk}")
    
    model_info = {
        'best_model': best_name,
        'best_auc': best_score,
        'model_comparison': results,
        'feature_names': list(X_train.columns),
        'risk_distribution': {
            'high': int(high_risk),
            'medium': int(medium_risk),
            'low': int(low_risk)
        }
    }
    
    if hasattr(best_model, 'feature_importances_'):
        importance_dict = dict(zip(X_train.columns, best_model.feature_importances_))
        model_info['feature_importance'] = importance_dict
        
        print(f"\nFeature Importance:")
        for feature, importance in sorted(importance_dict.items(), key=lambda x: x[1], reverse=True):
            print(f"{feature}: {importance:.3f}")
    
    with open('model_info/model_comparison.json', 'w') as f:
        json.dump(model_info, f, indent=2, default=str)
    
    return best_model, results

if __name__ == "__main__":
    train_models()