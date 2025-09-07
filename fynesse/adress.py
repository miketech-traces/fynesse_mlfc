import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

def prepare_data(nyeri_data, cambridge_data):
    """
    Prepare data for machine learning classification
    """
    # Add labels (0 for Nyeri, 1 for Cambridge)
    nyeri_data['city_label'] = 0
    cambridge_data['city_label'] = 1
    
    # Combine datasets
    combined_data = pd.concat([nyeri_data, cambridge_data], ignore_index=True)
    
    # Feature engineering (example)
    combined_data['lat_norm'] = (combined_data['geometry'].y - combined_data['geometry'].y.mean()) / combined_data['geometry'].y.std()
    combined_data['lon_norm'] = (combined_data['geometry'].x - combined_data['geometry'].x.mean()) / combined_data['geometry'].x.std()
    
    return combined_data

def train_model(data, features=['lat_norm', 'lon_norm'], target='city_label'):
    """
    Train a machine learning model to classify locations
    """
    # Prepare features and target
    X = data[features]
    y = data[target]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)
    
    # Evaluate model
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    
    print(f"Model Accuracy: {accuracy:.2f}")
    print("Confusion Matrix:")
    print(cm)
    
    return model, X_train, X_test, y_train, y_test
