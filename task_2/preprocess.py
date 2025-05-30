import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os

def load_and_clean_data():
    """Load and clean HR data"""
    print("Loading HR data...")
    
    df = pd.read_csv("merged_employee_data.csv")
    print(f"Loaded {len(df)} employee records")
    
    # Create target variable
    df['left'] = (df['EmployeeStatus'] == 'Terminated').astype(int)
    
    # Handle missing values
    df['Engagement Score'] = df['Engagement Score'].fillna(3)
    df['Satisfaction Score'] = df['Satisfaction Score'].fillna(3)
    df['Work-Life Balance Score'] = df['Work-Life Balance Score'].fillna(3)
    df['Current Employee Rating'] = df['Current Employee Rating'].fillna(3)
    df['Tenure'] = df['Tenure'].fillna(1)
    df['Performance Score'] = df['Performance Score'].fillna('Fully Meets')
    
    # Select required columns
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
    df_clean = df_clean.dropna()
    
    print(f"Clean dataset: {len(df_clean)} records")
    print(f"Attrition rate: {df_clean['left'].mean():.1%}")
    
    return df_clean

def prepare_training_data():
    """Prepare data for model training"""
    df = load_and_clean_data()
    
    # Encode performance scores
    encoder = LabelEncoder()
    df['Performance Score'] = encoder.fit_transform(df['Performance Score'])
    
    # Split features and target
    X = df.drop('left', axis=1)
    y = df['left']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Save processed data
    os.makedirs('data', exist_ok=True)
    X_train.to_csv('data/X_train.csv', index=False)
    X_test.to_csv('data/X_test.csv', index=False)
    y_train.to_csv('data/y_train.csv', index=False)
    y_test.to_csv('data/y_test.csv', index=False)
    
    # Save encoder
    joblib.dump(encoder, 'performance_encoder.pkl')
    
    print("Data preprocessing completed")
    return X_train, X_test, y_train, y_test

def get_data_summary():
    """Generate data summary for reporting"""
    df = load_and_clean_data()
    
    summary = {
        'total_records': len(df),
        'attrition_rate': df['left'].mean(),
        'avg_engagement': df['Engagement Score'].mean(),
        'avg_satisfaction': df['Satisfaction Score'].mean(),
        'avg_work_life_balance': df['Work-Life Balance Score'].mean(),
        'avg_tenure': df['Tenure'].mean(),
        'performance_distribution': df['Performance Score'].value_counts().to_dict()
    }
    
    return summary

if __name__ == "__main__":
    prepare_training_data()