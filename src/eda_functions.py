import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def box_plot(df: pd.DataFrame, column: str = None):
    """
    Plots a box plot for the entire DataFrame or a specified column.

    Parameters:
    df (pd.DataFrame): The input DataFrame.
    column (str): The name of the column to plot. If None, plots all numerical columns.

    Returns:
    None
    """
    plt.figure(figsize=(20, 10))

    if column:
        if pd.api.types.is_numeric_dtype(df[column]):
            sns.boxplot(data=df[column], orient='h', palette='Set3')
            plt.title(f'Box Plot of {column}')
        else:
            print(f"Column '{column}' is not numeric, so it cannot be plotted.")
    else:
        numeric_columns = df.select_dtypes(include=['number'])
        sns.boxplot(data=numeric_columns, orient='h', palette='Set3')
        plt.title('Box Plot of All Numeric Columns')

    plt.xlabel("Value")
    plt.tight_layout()
    plt.show()

def heat_map(df: pd.DataFrame):
    """
    Plots a heatmap of the correlation matrix for the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    None
    """
    numeric_df = df.select_dtypes(include=['number'])  # Select only numeric columns
    sns.set(style="whitegrid", font_scale=1)
    plt.figure(figsize=(15, 15))

    sns.heatmap(numeric_df.corr(), vmax=0.8, square=True, cmap="GnBu", linecolor="r", annot=True, annot_kws={'size': 10})
    
    plt.title('Correlation Heatmap')
    plt.tight_layout()
    plt.show()

def count_column_values(df, column_name, sort=False):
    """
    Counts the values in a specific column of a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    - column_name (str): The name of the column whose values will be counted.
    - sort (bool): If set to True, the results will be sorted in descending order of frequency.
    
    Returns:
    - pd.Series: A series with the unique values and their counts.
    """
    counts = df[column_name].value_counts()
    
    if sort:
        counts = counts.sort_values(ascending=False)
    
    return counts

def count_null_values(df):
    """
    Counts the number of missing (null) values in each column of a DataFrame and returns only columns with missing values.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    - pd.Series: A series with the column names and the number of missing values in each column, 
      including only those columns with at least one missing value.
    """
    null_counts = df.isnull().sum()
    # Filtra para conservar solo columnas con al menos un valor nulo
    null_counts_with_missing = null_counts[null_counts > 0]
    return null_counts_with_missing


def count_unique_values(df):
    """
    Returns a DataFrame with the number of unique values and data types for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame where the index is the column names and the columns include the number of unique values and the data type.
    """
    unique_counts = df.nunique()
    data_types = df.dtypes
    result = pd.DataFrame({
        'Unique Count': unique_counts,
        'Data Type': data_types
    })
    return result

def plot_histograms(df: pd.DataFrame, exclude_column: str = None):
    """
    Plots histograms for all numerical columns in the DataFrame.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    exclude_column (str): Optional column to exclude from the histogram plot (e.g., a categorical column).
    
    Returns:
    None
    """
    plt.figure(figsize=(15, 10))
    if exclude_column:
        df.drop([exclude_column], axis=1, errors='ignore').hist(figsize=(15, 10))
    else:
        df.hist(figsize=(15, 10))
    plt.show()

def plot_pairplot(df: pd.DataFrame, pairplot_columns: list, hue_column: str = None):
    """
    Plots pairplot for selected numerical columns, optionally with hue for a categorical column.
    
    Parameters:
    df (pd.DataFrame): The input DataFrame.
    pairplot_columns (list): List of numerical columns to include in the pairplot.
    hue_column (str): Optional categorical column for coloring in the pairplot.
    
    Returns:
    None
    """
    sns.pairplot(df, hue=hue_column, height=3, vars=pairplot_columns, kind='scatter', plot_kws={'s': 20})
    plt.show()
