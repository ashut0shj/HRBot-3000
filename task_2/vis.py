import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, roc_curve, auc
import json

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def create_feature_importance_plot():
    """Create feature importance visualization"""
    model = joblib.load('attrition_model.pkl')
    X_train = pd.read_csv('data/X_train.csv')
    
    feature_names = X_train.columns
    importance_scores = model.feature_importances_
    
    # Create horizontal bar plot
    plt.figure(figsize=(10, 6))
    y_pos = np.arange(len(feature_names))
    
    plt.barh(y_pos, importance_scores)
    plt.yticks(y_pos, feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Attrition Prediction Model')
    plt.tight_layout()
    
    plt.savefig('reports/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Feature importance plot saved")

def create_confusion_matrix_plot():
    """Create confusion matrix heatmap"""
    model = joblib.load('attrition_model.pkl')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    y_pred = model.predict(X_test)
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['Retained', 'Left'],
                yticklabels=['Retained', 'Left'])
    plt.title('Confusion Matrix - Attrition Prediction')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.tight_layout()
    
    plt.savefig('reports/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Confusion matrix plot saved")

def create_roc_curve_plot():
    """Create ROC curve"""
    model = joblib.load('attrition_model.pkl')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    y_pred_proba = model.predict_proba(X_test)
    
    # Fix IndexError by checking array shape
    if y_pred_proba.shape[1] > 1:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba[:, 0]
    
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba_pos)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Attrition Prediction Model')
    plt.legend(loc="lower right")
    plt.tight_layout()
    
    plt.savefig('reports/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("ROC curve plot saved")

def create_risk_distribution_plot():
    """Create risk score distribution plot"""
    model = joblib.load('attrition_model.pkl')
    X_test = pd.read_csv('data/X_test.csv')
    y_test = pd.read_csv('data/y_test.csv').squeeze()
    
    y_pred_proba = model.predict_proba(X_test)
    
    # Fix IndexError by checking array shape
    if y_pred_proba.shape[1] > 1:
        y_pred_proba_pos = y_pred_proba[:, 1]
    else:
        y_pred_proba_pos = y_pred_proba[:, 0]
    
    plt.figure(figsize=(10, 6))
    
    # Separate by actual outcome
    retained_scores = y_pred_proba_pos[y_test == 0]
    left_scores = y_pred_proba_pos[y_test == 1]
    
    plt.hist(retained_scores, bins=30, alpha=0.7, label='Retained Employees', 
             color='skyblue', density=True)
    plt.hist(left_scores, bins=30, alpha=0.7, label='Employees Who Left', 
             color='salmon', density=True)
    
    plt.axvline(x=0.4, color='orange', linestyle='--', label='Medium Risk Threshold')
    plt.axvline(x=0.7, color='red', linestyle='--', label='High Risk Threshold')
    
    plt.xlabel('Attrition Risk Score')
    plt.ylabel('Density')
    plt.title('Distribution of Attrition Risk Scores')
    plt.legend()
    plt.tight_layout()
    
    plt.savefig('reports/risk_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Risk distribution plot saved")

def create_correlation_matrix():
    """Create correlation matrix of features"""
    df = pd.read_csv('data/X_train.csv')
    
    plt.figure(figsize=(10, 8))
    correlation_matrix = df.corr()
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0,
                square=True, fmt='.2f')
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    
    plt.savefig('reports/correlation_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Correlation matrix plot saved")

def create_data_overview_plots():
    """Create overview plots of the dataset"""
    from preprocess import load_and_clean_data
    df = load_and_clean_data()
    
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Dataset Overview', fontsize=16)
    
    # Engagement score distribution
    axes[0, 0].hist(df['Engagement Score'], bins=20, alpha=0.7)
    axes[0, 0].set_title('Engagement Score Distribution')
    axes[0, 0].set_xlabel('Score')
    
    # Satisfaction score distribution
    axes[0, 1].hist(df['Satisfaction Score'], bins=20, alpha=0.7)
    axes[0, 1].set_title('Satisfaction Score Distribution')
    axes[0, 1].set_xlabel('Score')
    
    # Work-life balance distribution
    axes[0, 2].hist(df['Work-Life Balance Score'], bins=20, alpha=0.7)
    axes[0, 2].set_title('Work-Life Balance Distribution')
    axes[0, 2].set_xlabel('Score')
    
    # Tenure distribution
    axes[1, 0].hist(df['Tenure'], bins=20, alpha=0.7)
    axes[1, 0].set_title('Tenure Distribution')
    axes[1, 0].set_xlabel('Years')
    
    # Current rating distribution
    axes[1, 1].hist(df['Current Employee Rating'], bins=20, alpha=0.7)
    axes[1, 1].set_title('Employee Rating Distribution')
    axes[1, 1].set_xlabel('Rating')
    
    # Attrition rate
    attrition_counts = df['left'].value_counts()
    axes[1, 2].bar(['Retained', 'Left'], attrition_counts.values)
    axes[1, 2].set_title('Employee Retention vs Attrition')
    axes[1, 2].set_ylabel('Count')
    
    plt.tight_layout()
    plt.savefig('reports/data_overview.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("Data overview plots saved")

def generate_all_visualizations():
    """Generate all visualization files for reports"""
    import os
    os.makedirs('reports', exist_ok=True)
    
    print("Generating visualization reports...")
    
    create_feature_importance_plot()
    create_confusion_matrix_plot()
    create_roc_curve_plot()
    create_risk_distribution_plot()
    create_correlation_matrix()
    create_data_overview_plots()
    
    print("All visualizations saved in 'reports' folder")

def create_summary_report():
    """Create a comprehensive summary report"""
    from train_model import generate_model_report
    from preprocess import get_data_summary
    
    model_metrics = generate_model_report()
    data_summary = get_data_summary()
    
    report = {
        'project_title': 'Employee Attrition Prediction System',
        'data_summary': data_summary,
        'model_performance': {
            'accuracy': model_metrics['model_accuracy'],
            'total_test_samples': model_metrics['total_predictions']
        },
        'key_findings': {
            'top_predictive_factors': [factor[0] for factor in model_metrics['feature_importance'][:3]],
            'attrition_rate': data_summary['attrition_rate'],
            'model_accuracy': model_metrics['model_accuracy']
        },
        'visualizations_created': [
            'feature_importance.png',
            'confusion_matrix.png', 
            'roc_curve.png',
            'risk_distribution.png',
            'correlation_matrix.png',
            'data_overview.png'
        ]
    }
    
    with open('reports/summary_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("Summary report saved to reports/summary_report.json")
    return report

if __name__ == "__main__":
    generate_all_visualizations()
    create_summary_report()