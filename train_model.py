"""
Product Category Classification - Model Training Script

This script trains a machine learning model to automatically predict product categories
based on product titles. It loads the data, performs feature engineering, trains a
Logistic Regression model, and saves it for later use.

Usage:
    python train_model.py --data data/products.csv --output model.pkl

Author: Joyce Harvatovici
Date: December 2025
"""

import pandas as pd
import numpy as np
import pickle
import argparse
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')


def load_data(filepath):
    """Load the products dataset"""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"✓ Loaded {len(df)} products")
    print(f"✓ Found {df['category_label'].nunique()} categories")
    return df


def prepare_features(df):
    """Prepare features from the dataset"""
    print("\nPreparing features...")
    
    # Extract text features
    X = df['product_title']
    y = df['category_label']
    
    print(f"✓ Features prepared: {len(X)} samples")
    return X, y


def train_model(X_train, y_train):
    """Train the classification model"""
    print("\nTraining TF-IDF vectorizer...")
    
    # Create and fit TF-IDF vectorizer
    vectorizer = TfidfVectorizer(
        max_features=1000,
        ngram_range=(1, 2),
        min_df=2,
        lowercase=True
    )
    
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print(f"✓ TF-IDF vectorization complete: {X_train_tfidf.shape[1]} features")
    
    # Train Logistic Regression model
    print("\nTraining Logistic Regression model...")
    model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
    model.fit(X_train_tfidf, y_train)
    print("✓ Model training complete")
    
    return model, vectorizer


def evaluate_model(model, vectorizer, X_test, y_test):
    """Evaluate the trained model"""
    print("\n" + "="*70)
    print("MODEL EVALUATION")
    print("="*70)
    
    # Transform test data
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Make predictions
    y_pred = model.predict(X_test_tfidf)
    
    # Calculate accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nOverall Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    
    # Detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))
    
    return accuracy


def save_model(model, vectorizer, model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Save the trained model and vectorizer"""
    print("\nSaving model and vectorizer...")
    
    # Save model
    with open(model_path, 'wb') as f:
        pickle.dump(model, f)
    print(f"✓ Model saved to '{model_path}'")
    
    # Save vectorizer
    with open(vectorizer_path, 'wb') as f:
        pickle.dump(vectorizer, f)
    print(f"✓ Vectorizer saved to '{vectorizer_path}'")


def main():
    """Main training pipeline"""
    parser = argparse.ArgumentParser(description='Train product category classification model')
    parser.add_argument('--data', type=str, default='data/products.csv',
                        help='Path to the products CSV file')
    parser.add_argument('--output', type=str, default='model.pkl',
                        help='Path to save the trained model')
    parser.add_argument('--test-size', type=float, default=0.2,
                        help='Proportion of data to use for testing (default: 0.2)')
    
    args = parser.parse_args()
    
    print("="*70)
    print("PRODUCT CATEGORY CLASSIFICATION - MODEL TRAINING")
    print("="*70)
    
    # Load data
    df = load_data(args.data)
    
    # Prepare features
    X, y = prepare_features(df)
    
    # Split data
    print(f"\nSplitting data (test size: {args.test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=42,
        stratify=y
    )
    print(f"✓ Training set: {len(X_train)} samples")
    print(f"✓ Test set: {len(X_test)} samples")
    
    # Train model
    model, vectorizer = train_model(X_train, y_train)
    
    # Evaluate model
    accuracy = evaluate_model(model, vectorizer, X_test, y_test)
    
    # Save model
    save_model(model, vectorizer, args.output, 'vectorizer.pkl')
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print(f"\n✓ Final Model Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"✓ Model saved to: {args.output}")
    print(f"✓ Vectorizer saved to: vectorizer.pkl")
    print("\nYou can now use predict_category.py to make predictions!")


if __name__ == "__main__":
    main()
