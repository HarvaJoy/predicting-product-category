# 🏷️ Product Category Classification

**Automatic Product Categorization Using Machine Learning**

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.2%2B-orange)](https://scikit-learn.org/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)

## 📋 Project Overview

This project implements a machine learning solution to automatically classify products into categories based on their titles. The system is designed for e-commerce platforms that need to quickly and accurately categorize thousands of new products daily, reducing manual work and improving operational efficiency.

**Business Value:**
- ✅ Saves hours of manual categorization work
- ✅ Reduces human error in product classification
- ✅ Enables real-time product listing
- ✅ Scales efficiently with growing product catalogs

## 🎯 Problem Statement

E-commerce companies introduce thousands of new products every day. Each product must be correctly categorized, but manual classification is:
- Time-consuming
- Error-prone
- Difficult to scale

**Solution:** An intelligent ML model that automatically suggests the appropriate category based on the product title.

## 📊 Dataset

The project uses a comprehensive dataset (`products.csv`) with **38,000+ products** across **10 categories**:

- **Mobile Phones** (4,309 products)
- **Laptops** (4,974 products)
- **Headphones** (3,060 products)
- **Tablets** (4,227 products)
- **Smartwatches** (4,638 products)
- **Cameras** (3,301 products)
- **TVs** (4,372 products)
- **Gaming Consoles** (3,090 products)
- **Speakers** (3,432 products)
- **Computer Accessories** (3,486 products)

### Dataset Features:
- `product_id`: Unique identifier
- `product_title`: Product name (e.g., "Samsung Galaxy A52 128GB")
- `merchant_id`: Seller identifier
- `category_label`: Target category
- `product_code`: Internal product code
- `number_of_views`: Product view count
- `merchant_rating`: Seller rating
- `listing_date`: Date product was listed

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/HarvaJoy/predicting-product-category.git
   cd predicting-product-category
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Generate the dataset (if needed):**
   ```bash
   python generate_dataset.py
   ```

4. **Train the model:**
   ```bash
   python train_model.py --data data/products.csv --output model.pkl
   ```

5. **Test predictions:**
   ```bash
   python predict_category.py
   ```

## 💻 Usage

### Training the Model

```bash
# Basic training with default settings
python train_model.py

# Custom data path and output location
python train_model.py --data data/products.csv --output my_model.pkl

# Adjust test/train split
python train_model.py --test-size 0.25
```

### Making Predictions

**Interactive Mode:**
```bash
python predict_category.py
```

Then enter product titles when prompted:
```
➜ Samsung Galaxy S21 128GB
📦 Product: Samsung Galaxy S21 128GB
🏷️  Predicted Category: Mobile Phones
✨ Confidence: 98.5%
```

**Batch Mode:**
```bash
python predict_category.py --batch "iPhone 13 Pro" "MacBook Air M2" "Sony WH-1000XM5"
```

### Jupyter Notebook Analysis

Open and run the complete analysis notebook:
```bash
jupyter notebook analysis.ipynb
```

The notebook includes:
- Exploratory Data Analysis (EDA)
- Feature engineering experiments
- Model comparison and selection
- Performance evaluation with visualizations
- Confusion matrix analysis

## 📁 Project Structure

```
predicting-product-category/
├── README.md                   # This file
├── requirements.txt            # Python dependencies
├── generate_dataset.py         # Script to generate sample data
├── train_model.py             # Model training script
├── predict_category.py        # Prediction script (interactive/batch)
├── analysis.ipynb             # Complete Jupyter notebook analysis
├── data/
│   └── products.csv           # Product dataset
├── model.pkl                  # Trained model (generated after training)
├── vectorizer.pkl             # TF-IDF vectorizer (generated after training)
└── main.py                    # Task description and requirements
```

## 🔬 Methodology

### 1. **Data Exploration**
- Analyzed distribution of 38,889 products across 10 categories
- Examined product title patterns and lengths
- Identified key features for classification

### 2. **Feature Engineering**
- **Text Features:** TF-IDF vectorization (max 1000 features, bi-grams)
- **Numerical Features:** 
  - Word count in title
  - Character count
  - Presence of numbers (GB, model numbers)
  - Uppercase acronyms (USB, LED, HDMI)
  - Longest word length
  - Special character count

### 3. **Model Selection**
Compared multiple algorithms:
- **Naive Bayes:** Fast baseline, excellent for text
- **Logistic Regression:** ⭐ Best performer (95%+ accuracy)
- **Random Forest:** Good accuracy, more complex

### 4. **Evaluation Metrics**
- **Accuracy:** Overall correctness
- **Precision & Recall:** Per-category performance
- **F1-Score:** Balanced metric
- **Confusion Matrix:** Detailed misclassification analysis

## 📈 Results

### Model Performance

- **Algorithm:** Logistic Regression
- **Accuracy:** ~96% on test set
- **Training Time:** < 2 seconds
- **Prediction Time:** < 1ms per product

### Key Insights

✅ **High accuracy** across all product categories  
✅ **Consistent performance** - no single category performs poorly  
✅ **Fast predictions** - suitable for real-time applications  
✅ **Few misclassifications** - mostly between related categories

### Common Misclassifications
- Tablets ↔ Mobile Phones (similar naming patterns)
- Laptops ↔ Computer Accessories (monitors, keyboards)

## 🔧 Technical Details

### Technologies Used
- **Python 3.12**
- **pandas** - Data manipulation
- **scikit-learn** - Machine learning
- **matplotlib & seaborn** - Visualization
- **pickle** - Model serialization

### Model Architecture
```
Product Title (Text)
        ↓
TF-IDF Vectorization (1000 features, bi-grams)
        ↓
Logistic Regression Classifier
        ↓
Predicted Category + Confidence Score
```

## 🎓 Learning Outcomes

This project demonstrates:
- End-to-end ML project workflow
- Text classification with TF-IDF
- Model comparison and evaluation
- Production-ready code structure
- Clear documentation practices
- Git version control

## 🔮 Future Improvements

1. **Hyperparameter Tuning:** GridSearchCV for optimal parameters
2. **Deep Learning:** Try BERT or other transformer models
3. **Multi-feature Integration:** Include merchant rating, views, description
4. **Real-time API:** Flask/FastAPI REST endpoint
5. **Model Monitoring:** Track prediction confidence over time
6. **Active Learning:** Flag low-confidence predictions for review

## 👥 Author

**Joyce Harvatovici**
- GitHub: [@HarvaJoy](https://github.com/HarvaJoy)
- Project: Link Academy - Introduction to Machine Learning with Python

## 📝 License

This project is created for educational purposes as part of the Link Academy Machine Learning course.

## 🙏 Acknowledgments

- Link Academy for the comprehensive ML curriculum
- scikit-learn documentation and community
- Open-source Python data science ecosystem

---

**Note:** This project was developed as part of Task 3 for the Link Academy Machine Learning course (December 2025).
