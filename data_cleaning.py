"""
Data Cleaning Pipeline for Product Category Classification

This module provides functions to clean and preprocess the products dataset
for machine learning classification tasks.

Author: Joyce Harvatovici
Date: January 2026
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict
import re


def remove_missing_values(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Remove rows with missing values in critical columns"""
    if columns is None:
        columns = ['product_title', 'category_label']
    
    return df.dropna(subset=columns)


def remove_duplicates(df: pd.DataFrame, columns: List[str] = None) -> pd.DataFrame:
    """Remove duplicate products based on specified columns"""
    if columns is None:
        # Remove duplicates based on product_id or product_title
        if 'product_id' in df.columns:
            return df.drop_duplicates(subset=['product_id'], keep='first')
        else:
            return df.drop_duplicates(subset=['product_title'], keep='first')
    
    return df.drop_duplicates(subset=columns, keep='first')


def clean_product_titles(df: pd.DataFrame, column: str = 'product_title') -> pd.DataFrame:
    """Clean product titles by removing extra whitespace and standardizing format"""
    df_copy = df.copy()
    
    # Remove leading/trailing whitespace
    df_copy[column] = df_copy[column].str.strip()
    
    # Replace multiple spaces with single space
    df_copy[column] = df_copy[column].str.replace(r'\s+', ' ', regex=True)
    
    # Remove products with empty titles after cleaning
    df_copy = df_copy[df_copy[column].str.len() > 0]
    
    return df_copy


def standardize_categories(df: pd.DataFrame, column: str = 'category_label') -> pd.DataFrame:
    """Standardize category names"""
    df_copy = df.copy()
    
    # Strip whitespace from categories
    df_copy[column] = df_copy[column].str.strip()
    
    return df_copy


def filter_valid_categories(df: pd.DataFrame, 
                           valid_categories: List[str] = None,
                           column: str = 'category_label') -> pd.DataFrame:
    """Filter dataset to only include valid categories"""
    if valid_categories is None:
        # Get all unique categories from the dataset
        return df
    
    return df[df[column].isin(valid_categories)]


def remove_invalid_numeric_values(df: pd.DataFrame) -> pd.DataFrame:
    """Remove or fix invalid numeric values"""
    df_copy = df.copy()
    
    # Handle negative values in numeric columns
    numeric_columns = ['number_of_views', 'merchant_rating']
    
    for col in numeric_columns:
        if col in df_copy.columns:
            # Replace negative values with 0 or median
            if col == 'number_of_views':
                df_copy.loc[df_copy[col] < 0, col] = 0
            elif col == 'merchant_rating':
                # Merchant rating should be between 0 and 5
                df_copy = df_copy[(df_copy[col] >= 0) & (df_copy[col] <= 5)]
    
    return df_copy


def remove_short_titles(df: pd.DataFrame, 
                       column: str = 'product_title',
                       min_length: int = 3) -> pd.DataFrame:
    """Remove products with very short titles (likely invalid)"""
    df_copy = df.copy()
    df_copy = df_copy[df_copy[column].str.len() >= min_length]
    return df_copy


def get_cleaning_statistics(df_original: pd.DataFrame, 
                           df_cleaned: pd.DataFrame,
                           category_column: str = 'category_label') -> Dict:
    """Calculate and return cleaning statistics"""
    
    original_size = len(df_original)
    cleaned_size = len(df_cleaned)
    removed_rows = original_size - cleaned_size
    
    category_counts = df_cleaned[category_column].value_counts()
    
    # Calculate missing values before/after
    missing_before = df_original.isnull().sum().sum()
    missing_after = df_cleaned.isnull().sum().sum()
    
    stats = {
        'original_size': original_size,
        'cleaned_size': cleaned_size,
        'removed_rows': removed_rows,
        'removal_percentage': (removed_rows / original_size * 100) if original_size > 0 else 0,
        'category_counts': category_counts,
        'missing_values_before': missing_before,
        'missing_values_after': missing_after,
        'duplicate_check': original_size - df_original.drop_duplicates().shape[0]
    }
    
    return stats


def print_cleaning_statistics(stats: Dict) -> None:
    """Print detailed cleaning statistics"""
    original_size = stats['original_size']
    cleaned_size = stats['cleaned_size']
    removed_rows = stats['removed_rows']
    removal_percentage = stats['removal_percentage']
    category_counts = stats['category_counts']
    
    print("="*70)
    print("DATA CLEANING STATISTICS")
    print("="*70)
    
    print(f"\n📊 Dataset Size:")
    print(f"   Original: {original_size:,} products")
    print(f"   Cleaned:  {cleaned_size:,} products")
    print(f"   Removed:  {removed_rows:,} products ({removal_percentage:.2f}%)")
    
    print(f"\n📈 Category Distribution:")
    for category in category_counts.index:
        count = category_counts[category]
        percentage = (count / cleaned_size) * 100
        print(f"   {category:25s}: {count:5,} ({percentage:5.2f}%)")
    
    print(f"\n🔍 Data Quality:")
    print(f"   Missing values before: {stats['missing_values_before']}")
    print(f"   Missing values after:  {stats['missing_values_after']}")
    print(f"   Duplicates found:      {stats['duplicate_check']}")
    
    print("="*70)


def clean_product_data_pipeline(df: pd.DataFrame, 
                                columns_to_check: List[str] = None,
                                valid_categories: List[str] = None,
                                category_column: str = 'category_label',
                                min_title_length: int = 3,
                                verbose: bool = True) -> Tuple[pd.DataFrame, Dict]:
    """
    Complete data cleaning pipeline for product category dataset
    
    Args:
        df: Input DataFrame
        columns_to_check: Columns to check for missing values
        valid_categories: List of valid category names (None = all)
        category_column: Name of the category column
        min_title_length: Minimum length for product titles
        verbose: Print statistics
        
    Returns:
        Tuple of (cleaned DataFrame, statistics dictionary)
    """
    
    # Store original dataframe for statistics
    df_original = df.copy()
    
    if columns_to_check is None:
        columns_to_check = ['product_title', 'category_label']
    
    # Apply cleaning pipeline using pandas pipe
    df_cleaned = (df
                  .pipe(remove_missing_values, columns=columns_to_check)
                  .pipe(remove_duplicates)
                  .pipe(clean_product_titles, column='product_title')
                  .pipe(remove_short_titles, column='product_title', min_length=min_title_length)
                  .pipe(standardize_categories, column=category_column)
                  .pipe(filter_valid_categories, valid_categories=valid_categories, column=category_column)
                  .pipe(remove_invalid_numeric_values)
                  .reset_index(drop=True))
    
    # Calculate statistics
    stats = get_cleaning_statistics(df_original, df_cleaned, category_column)
    
    # Print statistics if verbose
    if verbose:
        print_cleaning_statistics(stats)
    
    return df_cleaned, stats

