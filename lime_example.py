"""
LIME (Local Interpretable Model-agnostic Explanations) Example
This is a beginner-friendly implementation demonstrating LIME for model interpretation.

Author: College Mini Project
Purpose: Educational demonstration of LIME for explaining machine learning predictions
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


def load_and_prepare_data():
    """
    Load the Iris dataset and prepare it for training.
    
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names, class_names
    """
    print("Loading Iris dataset...")
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    print(f"Dataset loaded successfully!")
    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")
    print(f"Features: {feature_names}")
    print(f"Classes: {class_names}")
    print("-" * 60)
    
    return X_train, X_test, y_train, y_test, feature_names, class_names


def train_model(X_train, y_train):
    """
    Train a Random Forest Classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        
    Returns:
        Trained model
    """
    print("Training Random Forest Classifier...")
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("Model trained successfully!")
    print("-" * 60)
    return model


def explain_prediction(model, X_train, X_test, feature_names, class_names, instance_idx=0):
    """
    Use LIME to explain a single prediction.
    
    Args:
        model: Trained machine learning model
        X_train: Training data
        X_test: Test data
        feature_names: List of feature names
        class_names: List of class names
        instance_idx: Index of the instance to explain
    """
    print(f"Explaining prediction for test instance {instance_idx}...")
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification'
    )
    
    # Select instance to explain
    instance = X_test[instance_idx]
    
    # Get model prediction
    prediction = model.predict([instance])[0]
    prediction_proba = model.predict_proba([instance])[0]
    
    print(f"\nInstance features:")
    for i, (fname, fval) in enumerate(zip(feature_names, instance)):
        print(f"  {fname}: {fval:.2f}")
    
    print(f"\nModel Prediction: {class_names[prediction]}")
    print(f"Prediction Probabilities:")
    for i, (cname, prob) in enumerate(zip(class_names, prediction_proba)):
        print(f"  {cname}: {prob:.4f}")
    
    # Generate explanation
    print("\nGenerating LIME explanation...")
    explanation = explainer.explain_instance(
        instance,
        model.predict_proba,
        num_features=len(feature_names)
    )
    
    print("\nLIME Explanation:")
    print("Top features contributing to the prediction:")
    for feature, weight in explanation.as_list():
        print(f"  {feature}: {weight:.4f}")
    
    # Save visualization
    print("\nSaving explanation visualization...")
    fig = explanation.as_pyplot_figure()
    plt.tight_layout()
    plt.savefig('lime_explanation.png', dpi=150, bbox_inches='tight')
    print("Visualization saved as 'lime_explanation.png'")
    plt.close()
    
    return explanation


def main():
    """
    Main function to run the LIME example.
    """
    print("=" * 60)
    print("LIME - Local Interpretable Model-agnostic Explanations")
    print("College Mini Project Demonstration")
    print("=" * 60)
    print()
    
    # Step 1: Load and prepare data
    X_train, X_test, y_train, y_test, feature_names, class_names = load_and_prepare_data()
    
    # Step 2: Train model
    model = train_model(X_train, y_train)
    
    # Step 3: Evaluate model
    train_score = model.score(X_train, y_train)
    test_score = model.score(X_test, y_test)
    print(f"Model Accuracy:")
    print(f"  Training: {train_score:.4f}")
    print(f"  Testing: {test_score:.4f}")
    print("-" * 60)
    print()
    
    # Step 4: Explain predictions using LIME
    # Explain first test instance
    explanation = explain_prediction(
        model, X_train, X_test, feature_names, class_names, instance_idx=0
    )
    
    print("\n" + "=" * 60)
    print("LIME explanation completed successfully!")
    print("=" * 60)
    
    return model, explanation


if __name__ == "__main__":
    main()
