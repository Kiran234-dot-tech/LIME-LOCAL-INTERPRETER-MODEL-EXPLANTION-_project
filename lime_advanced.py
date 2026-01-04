"""
Advanced LIME Example: Multiple Instance Explanations
This script demonstrates LIME on multiple test instances with comparison.

Author: College Mini Project
Purpose: Advanced demonstration of LIME with multiple predictions
"""

import numpy as np
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from lime.lime_tabular import LimeTabularExplainer
import matplotlib.pyplot as plt


def run_advanced_lime_demo():
    """
    Demonstrate LIME on multiple instances from the Wine dataset.
    """
    print("=" * 70)
    print("Advanced LIME Example: Wine Quality Classification")
    print("=" * 70)
    print()
    
    # Load Wine dataset
    print("Loading Wine dataset...")
    wine = load_wine()
    X = wine.data
    y = wine.target
    feature_names = wine.feature_names
    class_names = wine.target_names
    
    print(f"Dataset: {len(X)} samples, {len(feature_names)} features")
    print(f"Classes: {class_names}")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )
    
    # Train Gradient Boosting Classifier
    print("Training Gradient Boosting Classifier...")
    model = GradientBoostingClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    
    accuracy = model.score(X_test, y_test)
    print(f"Model Accuracy: {accuracy:.4f}")
    print("-" * 70)
    print()
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=class_names,
        mode='classification',
        random_state=42
    )
    
    # Explain multiple instances
    num_instances = 3
    print(f"Generating explanations for {num_instances} test instances...")
    print()
    
    for idx in range(num_instances):
        print(f"\n{'=' * 70}")
        print(f"Explanation {idx + 1}/{num_instances}")
        print('=' * 70)
        
        instance = X_test[idx]
        prediction = model.predict([instance])[0]
        proba = model.predict_proba([instance])[0]
        
        print(f"\nPredicted Class: {class_names[prediction]}")
        print(f"Confidence: {proba[prediction]:.4f}")
        
        # Generate explanation
        explanation = explainer.explain_instance(
            instance,
            model.predict_proba,
            num_features=5  # Show top 5 features
        )
        
        print("\nTop 5 Contributing Features:")
        for feature, weight in explanation.as_list():
            direction = "supports" if weight > 0 else "opposes"
            print(f"  • {feature}: {weight:+.4f} ({direction})")
        
        # Save individual explanation
        fig = explanation.as_pyplot_figure()
        plt.tight_layout()
        filename = f'lime_explanation_wine_{idx + 1}.png'
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"\nVisualization saved as '{filename}'")
        plt.close()
    
    print("\n" + "=" * 70)
    print("Advanced LIME demonstration completed!")
    print("=" * 70)
    
    # Create feature importance summary
    print("\nCreating feature importance summary...")
    create_feature_summary(model, feature_names, class_names)


def create_feature_summary(model, feature_names, class_names):
    """
    Create a summary of overall feature importance from the model.
    
    Args:
        model: Trained model with feature_importances_
        feature_names: List of feature names
        class_names: List of class names
    """
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        
        print("\nOverall Feature Importance (from the model):")
        print("-" * 70)
        for i, idx in enumerate(indices[:10]):
            print(f"{i+1:2d}. {feature_names[idx]:30s}: {importances[idx]:.4f}")
        
        # Create bar plot
        plt.figure(figsize=(10, 6))
        plt.bar(range(10), importances[indices[:10]])
        plt.xticks(range(10), [feature_names[i] for i in indices[:10]], rotation=45, ha='right')
        plt.xlabel('Features')
        plt.ylabel('Importance')
        plt.title('Top 10 Feature Importances (Global)')
        plt.tight_layout()
        plt.savefig('feature_importance_global.png', dpi=150, bbox_inches='tight')
        print("\nGlobal feature importance plot saved as 'feature_importance_global.png'")
        plt.close()


if __name__ == "__main__":
    run_advanced_lime_demo()
