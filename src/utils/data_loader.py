"""
Data loading utilities for NASA C-MAPSS turbofan dataset.

This module provides functions to load and validate the predictive maintenance
dataset files with proper error handling.
"""

import os
import pandas as pd
from typing import Tuple


def load_training_data(file_path: str) -> pd.DataFrame:
    """
    Load training data from NASA C-MAPSS dataset.
    
    Args:
        file_path (str): Path to PM_train.txt file
        
    Returns:
        pd.DataFrame: Training data with proper column names
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Training data file not found: {file_path}")
    
    try:
        # Read data with space separator
        df = pd.read_csv(file_path, sep=" ", header=None)
        
        # Drop empty columns (26, 27)
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)
        
        # Assign column names
        df.columns = [
            'id', 'cycle', 'setting1', 'setting2', 'setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
        ]
        
        # Sort by id and cycle for temporal consistency
        df = df.sort_values(['id', 'cycle'])
        
        print(f"Loaded training data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Number of engines: {df['id'].nunique()}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading training data: {str(e)}")


def load_test_data(file_path: str) -> pd.DataFrame:
    """
    Load test data from NASA C-MAPSS dataset.
    
    Args:
        file_path (str): Path to PM_test.txt file
        
    Returns:
        pd.DataFrame: Test data with proper column names
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Test data file not found: {file_path}")
    
    try:
        # Read data with space separator
        df = pd.read_csv(file_path, sep=" ", header=None)
        
        # Drop empty columns (26, 27)
        df.drop(df.columns[[26, 27]], axis=1, inplace=True)
        
        # Assign column names
        df.columns = [
            'id', 'cycle', 'setting1', 'setting2', 'setting3',
            's1', 's2', 's3', 's4', 's5', 's6', 's7', 's8', 's9', 's10',
            's11', 's12', 's13', 's14', 's15', 's16', 's17', 's18', 's19', 's20', 's21'
        ]
        
        print(f"Loaded test data: {df.shape[0]} rows, {df.shape[1]} columns")
        print(f"Number of engines: {df['id'].nunique()}")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading test data: {str(e)}")


def load_ground_truth(file_path: str) -> pd.DataFrame:
    """
    Load ground truth data (remaining cycles for test engines).
    
    Args:
        file_path (str): Path to PM_truth.txt file
        
    Returns:
        pd.DataFrame: Ground truth data with RUL values
        
    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If file format is invalid
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ground truth file not found: {file_path}")
    
    try:
        # Read ground truth data
        df = pd.read_csv(file_path, sep=" ", header=None)
        
        # Drop empty column
        df.drop(df.columns[[1]], axis=1, inplace=True)
        
        # Assign column name
        df.columns = ['RUL']
        
        # Add engine ID (1-indexed)
        df['id'] = df.index + 1
        
        print(f"Loaded ground truth: {df.shape[0]} engines")
        
        return df
        
    except Exception as e:
        raise ValueError(f"Error loading ground truth: {str(e)}")


def validate_data_consistency(train_df: pd.DataFrame, test_df: pd.DataFrame, 
                              truth_df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Validate consistency between training, test, and ground truth data.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        truth_df: Ground truth dataframe
        
    Returns:
        Tuple[bool, str]: (is_valid, message)
    """
    # Check column consistency
    if not set(train_df.columns) == set(test_df.columns):
        return False, "Training and test data have different columns"
    
    # Check ground truth matches test data
    if test_df['id'].nunique() != len(truth_df):
        return False, f"Ground truth count ({len(truth_df)}) doesn't match test engines ({test_df['id'].nunique()})"
    
    # Check for missing values
    if train_df.isnull().any().any():
        return False, "Training data contains missing values"
    
    if test_df.isnull().any().any():
        return False, "Test data contains missing values"
    
    return True, "Data validation passed"
