import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import joblib
import numpy as np

def train_attrition_model():
    """Train the attrition prediction model"""
    print("Training attrition prediction model...")
    
    # Load preprocessed data
    X_train = pd.read_csv('data/X_train.csv')
    X_test = pd.read_csv('data/X_test.csv')
    y_train = pd.read_csv('data/y_train.csv').squeeze()
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    # Initialize model with better regularization to prevent overfitting
    model = RandomForestClassifier(
        n_estimators=50,
        random_state=42,
        max_depth=5,
        min_samples_split=20,
        min_samples_leaf=10,
        max_features='sqrt'
    )
    
    # Train model
    model.fit(X_train, y_train)
    
    # Evaluate performance
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)
    
    train_accuracy = accuracy_score(y_train, train_pred)
    test_accuracy = accuracy_score(y_test, test_pred)
    
    print(f"Training accuracy: {train_accuracy:.3f}")
    print(f"Test accuracy: {test_accuracy:.3f}")
    
    # Save model
    joblib.dump(model, 'attrition_model.pkl')
    print("Model saved successfully")
    
    return model, test_accuracy

def get_feature_importance():
    """Get feature importance for reporting"""
    model = joblib.load('attrition_model.pkl')
    X_train = pd.read_csv('data/X_train.csv')
    
    feature_names = X_train.columns
    importance_scores = model.feature_importances_
    
    feature_importance = dict(zip(feature_names, importance_scores))
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
    
    return sorted_features

def test_model_prediction():
    """Test model with sample data"""
    print("Testing model with sample employee...")
    
    model = joblib.load('attrition_model.pkl')
    encoder = joblib.load('performance_encoder.pkl')
    X_train = pd.read_csv('data/X_train.csv')
    
    # Get available performance scores from encoder
    available_scores = encoder.classes_
    print(f"Available performance scores: {list(available_scores)}")
    
    # Use actual performance scores from the data
    performance_score = 'Fully Meets' if 'Fully Meets' in available_scores else available_scores[0]
    
    # Test cases using actual data structure with proper DataFrame
    test_cases = [
        {
            'name': 'High Risk Employee',
            'data': [1.5, 2.0, 2.0, 0.5, 2, encoder.transform([performance_score])[0]]
        },
        {
            'name': 'Medium Risk Employee', 
            'data': [3.0, 3.5, 3.0, 2.0, 3, encoder.transform([performance_score])[0]]
        },
        {
            'name': 'Low Risk Employee',
            'data': [4.5, 4.5, 4.0, 5.0, 4, encoder.transform([performance_score])[0]]
        }
    ]
    
    for case in test_cases:
        # Create DataFrame with proper feature names to avoid warning
        test_df = pd.DataFrame([case['data']], columns=X_train.columns)
        
        # Get prediction probabilities
        probabilities = model.predict_proba(test_df)[0]
        
        # Check if model predicts both classes (fix for IndexError)
        if len(probabilities) > 1:
            risk_prob = probabilities[1]
        else:
            risk_prob = probabilities[0]
            
        print(f"{case['name']}: {risk_prob:.3f} risk probability")

def generate_model_report():
    """Generate comprehensive model report"""
    model = joblib.load('attrition_model.pkl')
    
    # Load test data for evaluation
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)
    
    # Handle probability extraction based on number of classes
    if y_pred_proba.shape[1] > 1:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba[:, 0]
    
    # Metrics
    accuracy = accuracy_score(y_test, y_pred)
    conf_matrix = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Feature importance
    feature_importance = get_feature_importance()
    
    report = {
        'model_accuracy': accuracy,
        'confusion_matrix': conf_matrix.tolist(),
        'classification_report': class_report,
        'feature_importance': feature_importance,
        'total_predictions': len(y_test),
        'high_risk_threshold': 0.7,
        'medium_risk_threshold': 0.4
    }
    
    return report

if __name__ == "__main__":
    model, accuracy = train_attrition_model()
    test_model_prediction()
    
    # Generate and save report
    report = generate_model_report()
    import json
    with open('model_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("Model report saved to model_report.json")