import pandas as pd
import numpy as np

def check_data():
    df = pd.read_csv("dataset/merged_employee_data.csv")
    print("Data shape:", df.shape)
    print("\nEmployeeStatus values:")
    print(df['EmployeeStatus'].value_counts())
    
    terminated_statuses = ['Voluntarily Terminated', 'Terminated for Cause', 'Terminated']
    df['left'] = df['EmployeeStatus'].isin(terminated_statuses).astype(int)
    
    print(f"\nAttrition rate: {df['left'].mean():.2%}")
    print("Left distribution:")
    print(df['left'].value_counts())
    
    required_columns = [
        'Engagement Score', 
        'Satisfaction Score', 
        'Work-Life Balance Score',
        'Tenure', 
        'Current Employee Rating',
        'Performance Score'
    ]
    
    print(f"\nMissing values:")
    for col in required_columns:
        if col in df.columns:
            missing = df[col].isna().sum()
            print(f"{col}: {missing} ({missing/len(df)*100:.1f}%)")
        else:
            print(f"{col}: COLUMN NOT FOUND")
    
    print(f"\nPerformance Score values:")
    print(df['Performance Score'].value_counts(dropna=False))
    
    numeric_cols = ['Engagement Score', 'Satisfaction Score', 'Work-Life Balance Score', 'Tenure', 'Current Employee Rating']
    print(f"\nNumeric column stats:")
    for col in numeric_cols:
        if col in df.columns:
            col_data = pd.to_numeric(df[col], errors='coerce')
            print(f"{col}: min={col_data.min():.2f}, max={col_data.max():.2f}, mean={col_data.mean():.2f}")

if __name__ == "__main__":
    check_data()