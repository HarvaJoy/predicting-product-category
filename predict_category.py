"""
Product Category Prediction - Interactive Script

This script loads a trained machine learning model and allows users to 
interactively predict product categories based on product titles.

Usage:
    python predict_category.py
    
    Or specify custom model path:
    python predict_category.py --model path/to/model.pkl --vectorizer path/to/vectorizer.pkl

Author: Joyce Harvatovici
Date: December 2025
"""

import pickle
import argparse
import sys


def load_model(model_path='model.pkl', vectorizer_path='vectorizer.pkl'):
    """Load the trained model and vectorizer"""
    try:
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            vectorizer = pickle.load(f)
        
        return model, vectorizer
    except FileNotFoundError as e:
        print(f"Error: Could not find model files.")
        print(f"Please ensure '{model_path}' and '{vectorizer_path}' exist.")
        print(f"\nRun 'python train_model.py' first to train and save the model.")
        sys.exit(1)


def predict_category(product_title, model, vectorizer):
    """Predict the category for a given product title"""
    # Transform the product title
    product_tfidf = vectorizer.transform([product_title])
    
    # Predict the category
    predicted_category = model.predict(product_tfidf)[0]
    
    # Get confidence score if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(product_tfidf)[0]
        confidence = max(proba) * 100
    
    return predicted_category, confidence


def print_banner():
    """Print welcome banner"""
    print("="*70)
    print(" "*15 + "PRODUCT CATEGORY PREDICTOR")
    print("="*70)
    print("\nThis tool predicts the category of a product based on its title.")
    print("Enter a product title, and the model will suggest the best category.\n")


def print_examples():
    """Print example product titles"""
    print("\n" + "="*70)
    print("EXAMPLE PRODUCT TITLES YOU CAN TRY:")
    print("="*70)
    examples = [
        "Samsung Galaxy S21 128GB",
        "Apple MacBook Pro M2 Chip",
        "Sony WH-1000XM4 Wireless Headphones",
        "iPad Air 64GB WiFi",
        "LG 55 Inch 4K OLED TV",
        "Canon EOS R6 Mirrorless Camera",
        "Logitech MX Master 3 Mouse",
        "Nintendo Switch OLED",
        "JBL Flip 6 Bluetooth Speaker",
        "Apple Watch Series 8 GPS"
    ]
    for i, example in enumerate(examples, 1):
        print(f"{i:2d}. {example}")
    print("="*70 + "\n")


def interactive_mode(model, vectorizer):
    """Run interactive prediction mode"""
    print_banner()
    print_examples()
    
    while True:
        # Get user input
        print("\nEnter a product title (or 'quit' to exit, 'examples' to see examples):")
        product_title = input("➜ ").strip()
        
        # Check for exit command
        if product_title.lower() in ['quit', 'exit', 'q']:
            print("\nThank you for using the Product Category Predictor!")
            print("Goodbye! 👋\n")
            break
        
        # Show examples again
        if product_title.lower() == 'examples':
            print_examples()
            continue
        
        # Validate input
        if not product_title:
            print("⚠️  Please enter a valid product title.")
            continue
        
        # Make prediction
        try:
            predicted_category, confidence = predict_category(product_title, model, vectorizer)
            
            print("\n" + "-"*70)
            print(f"📦 Product: {product_title}")
            print(f"🏷️  Predicted Category: {predicted_category}")
            if confidence:
                print(f"✨ Confidence: {confidence:.1f}%")
            print("-"*70)
            
        except Exception as e:
            print(f"\n⚠️  Error making prediction: {e}")
            print("Please try again with a different product title.")


def batch_mode(products, model, vectorizer):
    """Predict categories for a list of products"""
    print_banner()
    print("Running in BATCH MODE\n")
    print("="*70)
    
    for i, product in enumerate(products, 1):
        predicted_category, confidence = predict_category(product, model, vectorizer)
        
        print(f"\n{i}. Product: {product}")
        print(f"   ➜ Category: {predicted_category}", end="")
        if confidence:
            print(f" (Confidence: {confidence:.1f}%)")
        else:
            print()
    
    print("\n" + "="*70)


def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Predict product categories from titles')
    parser.add_argument('--model', type=str, default='model.pkl',
                        help='Path to the trained model file')
    parser.add_argument('--vectorizer', type=str, default='vectorizer.pkl',
                        help='Path to the TF-IDF vectorizer file')
    parser.add_argument('--batch', nargs='+', 
                        help='Batch mode: provide product titles as arguments')
    
    args = parser.parse_args()
    
    # Load model and vectorizer
    print("Loading model and vectorizer...")
    model, vectorizer = load_model(args.model, args.vectorizer)
    print("✓ Model loaded successfully!\n")
    
    # Run in batch or interactive mode
    if args.batch:
        batch_mode(args.batch, model, vectorizer)
    else:
        interactive_mode(model, vectorizer)


if __name__ == "__main__":
    main()
