"""
Data preprocessing utilities for predictive maintenance.

This module contains functions for RUL calculation, normalization,
and sequence generation for LSTM models.
"""

import numpy as np
import pandas as pd
from sklearn import preprocessing
from typing import Tuple, List


def calculate_rul(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate Remaining Useful Life (RUL) for each engine cycle.
    
    RUL = max_cycle - current_cycle for each engine
    
    Args:
        df (pd.DataFrame): Dataframe with 'id' and 'cycle' columns
        
    Returns:
        pd.DataFrame: Dataframe with added 'RUL' column
    """
    # Get maximum cycle for each engine
    rul = pd.DataFrame(df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    
    # Merge and calculate RUL
    df = df.merge(rul, on=['id'], how='left')
    df['RUL'] = df['max'] - df['cycle']
    df.drop('max', axis=1, inplace=True)
    
    return df


def generate_failure_labels(df: pd.DataFrame, w0: int = 15, w1: int = 30) -> pd.DataFrame:
    """
    Generate binary and multi-class failure labels based on RUL windows.
    
    Args:
        df (pd.DataFrame): Dataframe with 'RUL' column
        w0 (int): Early warning window (cycles)
        w1 (int): Critical failure window (cycles)
        
    Returns:
        pd.DataFrame: Dataframe with added 'label1' and 'label2' columns
        
    Notes:
        - label1: Binary (0 = safe, 1 = will fail within w1 cycles)
        - label2: Multi-class (0 = safe, 1 = warning, 2 = critical)
    """
    # Binary classification label
    df['label1'] = np.where(df['RUL'] <= w1, 1, 0)
    
    # Multi-class label
    df['label2'] = df['label1']
    df.loc[df['RUL'] <= w0, 'label2'] = 2
    
    return df


def normalize_features(train_df: pd.DataFrame, test_df: pd.DataFrame = None,
                       method: str = 'minmax') -> Tuple[pd.DataFrame, pd.DataFrame, object]:
    """
    Normalize features using specified method.
    
    Args:
        train_df (pd.DataFrame): Training dataframe
        test_df (pd.DataFrame, optional): Test dataframe
        method (str): Normalization method ('minmax', 'standard', 'robust')
        
    Returns:
        Tuple containing:
            - Normalized training dataframe
            - Normalized test dataframe (if provided)
            - Fitted scaler object
    """
    # Add normalized cycle column
    train_df['cycle_norm'] = train_df['cycle']
    
    # Columns to normalize (exclude id, cycle, RUL, labels)
    cols_normalize = train_df.columns.difference(['id', 'cycle', 'RUL', 'label1', 'label2'])
    
    # Select scaler
    if method == 'minmax':
        scaler = preprocessing.MinMaxScaler()
    elif method == 'standard':
        scaler = preprocessing.StandardScaler()
    elif method == 'robust':
        scaler = preprocessing.RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform training data
    norm_train_df = pd.DataFrame(
        scaler.fit_transform(train_df[cols_normalize]),
        columns=cols_normalize,
        index=train_df.index
    )
    
    # Join normalized columns back
    join_df = train_df[train_df.columns.difference(cols_normalize)].join(norm_train_df)
    train_df = join_df.reindex(columns=train_df.columns)
    
    # Transform test data if provided
    if test_df is not None:
        test_df['cycle_norm'] = test_df['cycle']
        
        norm_test_df = pd.DataFrame(
            scaler.transform(test_df[cols_normalize]),
            columns=cols_normalize,
            index=test_df.index
        )
        
        test_join_df = test_df[test_df.columns.difference(cols_normalize)].join(norm_test_df)
        test_df = test_join_df.reindex(columns=test_df.columns)
        test_df = test_df.reset_index(drop=True)
        
        return train_df, test_df, scaler
    
    return train_df, None, scaler


def generate_sequences(id_df: pd.DataFrame, seq_length: int, seq_cols: List[str]) -> np.ndarray:
    """
    Generate sliding window sequences from time series data.
    
    Args:
        id_df (pd.DataFrame): Dataframe for single engine
        seq_length (int): Sequence length (time steps)
        seq_cols (List[str]): Column names to include in sequences
        
    Yields:
        np.ndarray: Sequence of shape (seq_length, n_features)
        
    Notes:
        Only sequences that meet the window length are considered.
        No padding is used for shorter sequences.
    """
    data_matrix = id_df[seq_cols].values
    num_elements = data_matrix.shape[0]
    
    # Generate sliding windows
    for start, stop in zip(range(0, num_elements - seq_length), 
                           range(seq_length, num_elements)):
        yield data_matrix[start:stop, :]


def generate_labels(id_df: pd.DataFrame, seq_length: int, label_col: str) -> np.ndarray:
    """
    Generate labels corresponding to sequences.
    
    Args:
        id_df (pd.DataFrame): Dataframe for single engine
        seq_length (int): Sequence length (time steps)
        label_col (str): Label column name
        
    Returns:
        np.ndarray: Labels for each sequence
        
    Notes:
        The first seq_length labels are discarded because the first sequence
        uses them as input. Subsequent sequences get one label each.
    """
    data_matrix = id_df[label_col].values
    num_elements = data_matrix.shape[0]
    
    # Return labels starting from seq_length position
    return data_matrix[seq_length:num_elements, :]


def prepare_sequences_and_labels(df: pd.DataFrame, seq_length: int, 
                                 label_col: str = 'RUL') -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare sequences and labels for all engines in dataframe.
    
    Args:
        df (pd.DataFrame): Dataframe with multiple engines
        seq_length (int): Sequence length (time steps)
        label_col (str): Label column name ('RUL' or 'label1')
        
    Returns:
        Tuple containing:
            - Sequence array of shape (n_samples, seq_length, n_features)
            - Label array of shape (n_samples, 1)
    """
    # Define feature columns
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    
    # Generate sequences for all engines
    seq_gen = (list(generate_sequences(df[df['id'] == engine_id], seq_length, sequence_cols))
               for engine_id in df['id'].unique())
    
    seq_array = np.concatenate(list(seq_gen)).astype(np.float32)
    
    # Generate corresponding labels
    label_gen = [generate_labels(df[df['id'] == engine_id], seq_length, [label_col])
                 for engine_id in df['id'].unique()]
    
    label_array = np.concatenate(label_gen).astype(np.float32)
    
    print(f"Generated sequences: {seq_array.shape}")
    print(f"Generated labels: {label_array.shape}")
    
    return seq_array, label_array


def prepare_test_sequences(test_df: pd.DataFrame, seq_length: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Prepare test sequences by taking the last sequence for each engine.
    
    Args:
        test_df (pd.DataFrame): Test dataframe
        seq_length (int): Sequence length (time steps)
        
    Returns:
        Tuple containing:
            - Sequence array of shape (n_engines, seq_length, n_features)
            - Mask array indicating which engines have sufficient data
    """
    # Define feature columns
    sensor_cols = ['s' + str(i) for i in range(1, 22)]
    sequence_cols = ['setting1', 'setting2', 'setting3', 'cycle_norm']
    sequence_cols.extend(sensor_cols)
    
    # Extract last sequence for each engine (only if sufficient length)
    seq_array_test_last = [
        test_df[test_df['id'] == engine_id][sequence_cols].values[-seq_length:]
        for engine_id in test_df['id'].unique()
        if len(test_df[test_df['id'] == engine_id]) >= seq_length
    ]
    
    seq_array_test_last = np.asarray(seq_array_test_last).astype(np.float32)
    
    # Create mask for engines with sufficient data
    y_mask = [len(test_df[test_df['id'] == engine_id]) >= seq_length 
              for engine_id in test_df['id'].unique()]
    
    print(f"Test sequences: {seq_array_test_last.shape}")
    print(f"Engines with sufficient data: {sum(y_mask)}/{len(y_mask)}")
    
    return seq_array_test_last, np.array(y_mask)


def preprocess_data(train_df: pd.DataFrame, test_df: pd.DataFrame, truth_df: pd.DataFrame,
                   w0: int = 15, w1: int = 30, normalization: str = 'minmax') -> dict:
    """
    Complete preprocessing pipeline for training and test data.
    
    Args:
        train_df: Training dataframe
        test_df: Test dataframe
        truth_df: Ground truth dataframe
        w0: Early warning window
        w1: Critical failure window
        normalization: Normalization method
        
    Returns:
        dict: Dictionary containing all preprocessed data and scaler
    """
    # Process training data
    train_df = calculate_rul(train_df)
    train_df = generate_failure_labels(train_df, w0, w1)
    
    # Process test data with ground truth
    rul = pd.DataFrame(test_df.groupby('id')['cycle'].max()).reset_index()
    rul.columns = ['id', 'max']
    truth_df.columns = ['more']
    truth_df['id'] = truth_df.index + 1
    truth_df['max'] = rul['max'] + truth_df['more']
    truth_df.drop('more', axis=1, inplace=True)
    
    test_df = test_df.merge(truth_df, on=['id'], how='left')
    test_df['RUL'] = test_df['max'] - test_df['cycle']
    test_df.drop('max', axis=1, inplace=True)
    test_df = generate_failure_labels(test_df, w0, w1)
    
    # Normalize features
    train_df, test_df, scaler = normalize_features(train_df, test_df, normalization)
    
    return {
        'train_df': train_df,
        'test_df': test_df,
        'scaler': scaler
    }
