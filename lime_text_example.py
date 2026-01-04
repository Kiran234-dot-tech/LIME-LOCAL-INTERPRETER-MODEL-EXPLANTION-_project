"""
LIME Text Classification Example
This script demonstrates LIME for text classification.

Author: College Mini Project
Purpose: Educational demonstration of LIME for text data
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline
from lime.lime_text import LimeTextExplainer
import numpy as np


def create_sample_dataset():
    """
    Create a simple dataset for sentiment classification.
    
    Returns:
        tuple: (texts, labels)
    """
    # Simple movie review dataset
    texts = [
        "This movie was fantastic! I loved every minute of it.",
        "Absolutely brilliant performance by the actors.",
        "One of the best films I have ever seen.",
        "Amazing storyline and great direction.",
        "Wonderful cinematography and excellent script.",
        "This was terrible. Complete waste of time.",
        "Very disappointing and boring movie.",
        "Worst film I have ever watched.",
        "Poor acting and weak storyline.",
        "Absolutely horrible. Do not watch this.",
        "The movie was okay, nothing special.",
        "It was decent but could have been better.",
        "Average film with some good moments.",
        "Not bad but not great either.",
    ]
    
    # 0 = Negative, 1 = Positive, 2 = Neutral
    labels = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 2, 2, 2, 2]
    
    return texts, labels


def run_text_lime_demo():
    """
    Demonstrate LIME for text classification.
    """
    print("=" * 70)
    print("LIME Text Classification Example")
    print("=" * 70)
    print()
    
    # Create dataset
    texts, labels = create_sample_dataset()
    class_names = ['Negative', 'Positive', 'Neutral']
    
    print("Dataset created:")
    print(f"  Total samples: {len(texts)}")
    print(f"  Classes: {class_names}")
    print()
    
    # Create a simple text classification pipeline
    print("Training Naive Bayes text classifier...")
    model = make_pipeline(
        TfidfVectorizer(),
        MultinomialNB()
    )
    model.fit(texts, labels)
    print("Model trained successfully!")
    print("-" * 70)
    print()
    
    # Create LIME text explainer
    explainer = LimeTextExplainer(class_names=class_names)
    
    # Test instances to explain
    test_texts = [
        "This movie is absolutely wonderful and amazing!",
        "Terrible film, very boring and disappointing.",
        "The movie was fine, nothing extraordinary.",
    ]
    
    print("Explaining predictions for test samples...")
    print()
    
    for idx, text in enumerate(test_texts):
        print(f"\n{'=' * 70}")
        print(f"Example {idx + 1}/{len(test_texts)}")
        print('=' * 70)
        print(f"\nText: \"{text}\"")
        
        # Get prediction
        prediction = model.predict([text])[0]
        proba = model.predict_proba([text])[0]
        
        print(f"\nPredicted Class: {class_names[prediction]}")
        print(f"Confidence: {proba[prediction]:.4f}")
        
        print("\nPrediction Probabilities:")
        for i, (cname, prob) in enumerate(zip(class_names, proba)):
            print(f"  {cname}: {prob:.4f}")
        
        # Generate LIME explanation
        print("\nGenerating LIME explanation...")
        explanation = explainer.explain_instance(
            text,
            model.predict_proba,
            num_features=6
        )
        
        print("\nWords contributing to the prediction:")
        print("(Positive values support the prediction, negative oppose it)")
        for word, weight in explanation.as_list():
            direction = "supports" if weight > 0 else "opposes"
            print(f"  • '{word}': {weight:+.4f} ({direction})")
        
        # Save HTML visualization
        html_filename = f'lime_text_explanation_{idx + 1}.html'
        explanation.save_to_file(html_filename)
        print(f"\nHTML visualization saved as '{html_filename}'")
    
    print("\n" + "=" * 70)
    print("Text classification LIME demonstration completed!")
    print("=" * 70)


if __name__ == "__main__":
    run_text_lime_demo()
