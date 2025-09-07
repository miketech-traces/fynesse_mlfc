from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from assess import prepare_data, CITIES_KENYA, CITIES_ENGLAND

def run_model():
    """
    Trains a classification model and makes predictions on test data.
    This function represents the "Address" part of the pipeline.
    """
    # Combine and split city data for a demonstration
    cities_df_kenya = pd.DataFrame.from_dict(CITIES_KENYA, orient="index")
    cities_df_kenya["country"] = "Kenya"

    cities_df_england = pd.DataFrame.from_dict(CITIES_ENGLAND, orient="index")
    cities_df_england["country"] = "England"

    df_full = pd.concat([cities_df_kenya, cities_df_england])

    # Manually define a train-test split for this example
    train_cities = df_full.index[:4]
    test_cities = df_full.index[4:]

    df_train = df_full.loc[train_cities]
    df_test = df_full.loc[test_cities]

    # Prepare data using the function from assess.py
    X_train, y_train = prepare_data(df_train)
    X_test, y_test = prepare_data(df_test)

    # Train the Decision Tree Classifier
    clf = DecisionTreeClassifier(random_state=42)
    print("\nTraining the Decision Tree model...")
    clf.fit(X_train, y_train)

    # Make predictions
    print("\nMaking predictions on the test cities...")
    y_pred = clf.predict(X_test)

    # Display results
    print("\nPredictions for the test cities:")
    print(pd.Series(y_pred, index=X_test.index))

if __name__ == "__main__":
    run_model()
