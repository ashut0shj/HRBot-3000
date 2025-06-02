import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, RobustScaler
from sklearn.model_selection import train_test_split
import joblib
import os

def prepare_training_data():
    df = pd.read_csv("dataset/merged_employee_data.csv")
    
    terminated_statuses = ['Voluntarily Terminated', 'Terminated for Cause', 'Terminated']
    df['left'] = df['EmployeeStatus'].isin(terminated_statuses).astype(int)
    
    required_columns = [
        'Engagement Score', 
        'Satisfaction Score', 
        'Work-Life Balance Score',
        'Tenure', 
        'Current Employee Rating',
        'Performance Score',
        'left'
    ]
    
    df_clean = df[required_columns].copy()
    
    numeric_cols = ['Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score', 
                   'Tenure', 'Current Employee Rating']
    
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    df_clean['Performance Score'] = df_clean['Performance Score'].fillna(df_clean['Performance Score'].mode()[0])
    
    df_clean = df_clean.dropna()
    
    if df_clean['left'].sum() == 0:
        print("ERROR: No attrition cases found in data")
        return None
    
    encoder = LabelEncoder()
    df_clean['Performance Score'] = encoder.fit_transform(df_clean['Performance Score'])
    
    scaler = RobustScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])
    
    X = df_clean.drop('left', axis=1)
    y = df_clean['left']
    
    print(f"Total samples: {len(df_clean)}")
    print(f"Attrition cases: {y.sum()}")
    print(f"Attrition rate: {y.mean():.2%}")
    
    if y.sum() < 10:
        print("WARNING: Very few attrition cases, model may not be reliable")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    joblib.dump(encoder, 'performance_encoder.pkl')
    joblib.dump(scaler, 'scaler.pkl')
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    prepare_training_data()