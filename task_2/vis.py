import matplotlib.pyplot as plt
import pandas as pd
import joblib
import os

def create_feature_importance_plot():
    model = joblib.load('attrition_model.pkl')
    X_train = pd.read_csv('data/X_train.csv')
    
    feature_names = X_train.columns
    importance_scores = model.feature_importances_
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(len(feature_names)), importance_scores)
    plt.yticks(range(len(feature_names)), feature_names)
    plt.xlabel('Feature Importance')
    plt.title('Feature Importance in Attrition Prediction')
    
    os.makedirs('charts', exist_ok=True)
    plt.savefig('charts/feature_importance.png')
    plt.close()

def create_data_distribution_plot():
    df = pd.read_csv("dataset/merged_employee_data.csv")
    df['left'] = (df['EmployeeStatus'] == 'Terminated').astype(int)
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 3, 1)
    plt.hist(df['Engagement Score'].dropna(), bins=20)
    plt.title('Engagement Score')
    
    plt.subplot(2, 3, 2)
    plt.hist(df['Satisfaction Score'].dropna(), bins=20)
    plt.title('Satisfaction Score')
    
    plt.subplot(2, 3, 3)
    plt.hist(df['Work-Life Balance Score'].dropna(), bins=20)
    plt.title('Work-Life Balance')
    
    plt.subplot(2, 3, 4)
    plt.hist(df['Tenure'].dropna(), bins=20)
    plt.title('Tenure')
    
    plt.subplot(2, 3, 5)
    plt.hist(df['Current Employee Rating'].dropna(), bins=20)
    plt.title('Employee Rating')
    
    plt.subplot(2, 3, 6)
    attrition_counts = df['left'].value_counts()
    plt.bar(['Retained', 'Left'], attrition_counts.values)
    plt.title('Retention vs Attrition')
    
    plt.tight_layout()
    plt.savefig('charts/data_distribution.png')
    plt.close()

if __name__ == "__main__":
    create_feature_importance_plot()
    create_data_distribution_plot()
    print("Charts saved to 'charts' folder")