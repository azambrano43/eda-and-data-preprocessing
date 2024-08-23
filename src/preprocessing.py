import pandas as pd
import numpy as np


def clean_airlines(df, airlines_to_remove):
    df_filt = df[~df['Airline'].isin(airlines_to_remove)].dropna()
    return df_filt

def drop_null_values(df, column):
    return df[df[column].notna()]

def convert_columns(df: pd.DataFrame, columns: list, dtype: str) -> pd.DataFrame:
    """
    Converts specified columns in the DataFrame to the given data type.
    Creates a copy of the DataFrame to avoid modifying the original.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to be converted.
    dtype (str): Desired data type for the columns. Options: 'String', 'Bool', 'Int', 'Float', 'Date'.

    Returns:
    pd.DataFrame: A new DataFrame with specified columns converted to the desired type.
    """
    # Create a copy of the DataFrame to avoid modifying the original
    df_copy = df.copy()
    
    # Ensure `dtype` is a valid type
    if dtype not in ['String', 'Bool', 'Int', 'Float', 'Date']:
        raise ValueError(f"Unsupported dtype '{dtype}'. Use 'String', 'Bool', 'Int', 'Float', or 'Date'.")

    for col in columns:
        if col not in df_copy.columns:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
        
        if dtype == 'String':
            df_copy[col] = df_copy[col].astype("string")  # Convert to string
        elif dtype == 'Bool':
            df_copy[col] = df_copy[col].astype("string").str.lower().map({'true': True, 'false': False, '1': True, '0': False})
        elif dtype == 'Int':
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('Int64')  # Use 'Int64' to handle NaN
        elif dtype == 'Float':
            df_copy[col] = pd.to_numeric(df_copy[col], errors='coerce').astype('float')
        elif dtype == 'Date':
            df_copy[col] = pd.to_datetime(df_copy[col], errors='coerce')  # Convert to datetime, handling errors with coercion
    
    return df_copy


def get_object_columns(df: pd.DataFrame) -> list:
    """
    Returns a list of column names in the DataFrame that are of type 'object'.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    list: List of column names with data type 'object'.
    """
    object_columns = df.select_dtypes(include=['object']).columns.tolist()
    return object_columns



