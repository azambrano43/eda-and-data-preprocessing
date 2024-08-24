import pandas as pd
import numpy as np
import warnings
from sklearn.preprocessing import LabelEncoder


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

def encode_columns(df: pd.DataFrame, columns: list, encoding_type: str) -> pd.DataFrame:
    """
    Encodes specified columns using either LabelEncoder or pd.get_dummies based on the user's choice.
    Creates a copy of the DataFrame to avoid modifying the original.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to be encoded.
    encoding_type (str): Encoding type, either 'label' for LabelEncoder or 'dummies' for pd.get_dummies.

    Returns:
    pd.DataFrame: A new DataFrame with the specified columns encoded.
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()

    if encoding_type == 'label':
        # Apply LabelEncoder to each column in the columns list
        label_encoder = LabelEncoder()
        for col in columns:
            if col in df_copy.columns:
                df_copy[col] = label_encoder.fit_transform(df_copy[col].astype(str))
            else:
                raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    elif encoding_type == 'dummies':
        # Apply pd.get_dummies to the specified columns
        df_copy = pd.get_dummies(df_copy, columns=columns, drop_first=True)
    
    else:
        raise ValueError("Invalid encoding type. Use 'label' for LabelEncoder or 'dummies' for pd.get_dummies.")
    
    return df_copy

def remove_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Removes specified columns from the DataFrame without modifying the original.
    Creates a copy of the DataFrame and returns it with the specified columns removed.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to be removed.

    Returns:
    pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Remove the specified columns
    df_copy = df_copy.drop(columns=columns, errors='ignore')
    
    return df_copy

def fill_missing_values(df: pd.DataFrame, columns: list, method: str) -> pd.DataFrame:
    """
    Fills missing values in the specified columns of the DataFrame using mean, median, or mode.
    - Mode can only be used for integer columns. String columns will trigger a warning.
    - If a string column is detected, a warning is issued.
    Creates a copy of the DataFrame to avoid modifying the original.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names where missing values should be filled.
    method (str): Method to fill missing values, either 'mean', 'median', or 'mode'. 

    Returns:
    pd.DataFrame: A new DataFrame with missing values filled in the specified columns.
    """
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Validate method
    if method not in ['mean', 'median', 'mode']:
        raise ValueError("Invalid method. Use 'mean', 'median', or 'mode'.")
    
    # Fill missing values
    for col in columns:
        if col in df_copy.columns:
            col_dtype = df_copy[col].dtype
            
            if pd.api.types.is_string_dtype(col_dtype):
                warnings.warn(f"Column '{col}' is of type string. Please encode string columns before using this function.")
                continue
            
            if method == 'mean':
                # Mean can be applied to numeric columns (int or float)
                if pd.api.types.is_numeric_dtype(col_dtype):
                    mean_value = df_copy[col].mean()
                    # If the column is integer type, round the mean and convert it to int
                    if pd.api.types.is_integer_dtype(col_dtype):
                        df_copy[col].fillna(round(mean_value), inplace=True)
                    else:
                        df_copy[col].fillna(mean_value, inplace=True)
                else:
                    raise ValueError(f"Cannot apply mean to non-numeric column '{col}'.")
            
            elif method == 'median':
                # Median can be applied to numeric columns (int or float)
                if pd.api.types.is_numeric_dtype(col_dtype):
                    median_value = df_copy[col].median()
                    # If the column is integer type, convert the median to int
                    if pd.api.types.is_integer_dtype(col_dtype):
                        df_copy[col].fillna(int(median_value), inplace=True)
                    else:
                        df_copy[col].fillna(median_value, inplace=True)
                else:
                    raise ValueError(f"Cannot apply median to non-numeric column '{col}'.")
            
            elif method == 'mode':
                # Mode can only be applied to integer columns
                if pd.api.types.is_integer_dtype(col_dtype):
                    mode_value = df_copy[col].mode()[0]
                    df_copy[col].fillna(mode_value, inplace=True)
                else:
                    warnings.warn(f"Mode can only be applied to integer columns. Column '{col}' will be skipped.")
        else:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    return df_copy

def clip_outliers(df: pd.DataFrame, columns: list, lower_percentile: float, upper_percentile: float) -> pd.DataFrame:
    """
    Clips outliers in the specified columns of the DataFrame based on user-defined percentiles.
    Values below the lower percentile are set to the lower percentile value.
    Values above the upper percentile are set to the upper percentile value.
    Creates a copy of the DataFrame to avoid modifying the original.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    columns (list): List of column names to apply outlier clipping.
    lower_percentile (float): Lower percentile for clipping (e.g., 0.10 for the 10th percentile).
    upper_percentile (float): Upper percentile for clipping (e.g., 0.90 for the 90th percentile).

    Returns:
    pd.DataFrame: A new DataFrame with outliers clipped in the specified columns.
    """
    # Validate percentiles
    if not (0 <= lower_percentile < upper_percentile <= 1):
        raise ValueError("Percentiles must be between 0 and 1, with lower_percentile < upper_percentile.")
    
    # Create a copy of the DataFrame
    df_copy = df.copy()
    
    # Clip outliers
    for col in columns:
        if col in df_copy.columns:
            if pd.api.types.is_numeric_dtype(df_copy[col]):
                lower_value = df_copy[col].quantile(lower_percentile)
                upper_value = df_copy[col].quantile(upper_percentile)
                df_copy[col] = df_copy[col].clip(lower=lower_value, upper=upper_value)
            else:
                raise ValueError(f"Column '{col}' is not numeric. Outlier clipping only applies to numeric columns.")
        else:
            raise ValueError(f"Column '{col}' does not exist in the DataFrame.")
    
    return df_copy