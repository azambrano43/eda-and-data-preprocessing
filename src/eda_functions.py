import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

def box_plot(df, column=None):
    if column is None:
        plt.figure(figsize=(20, 20))
        sns.boxplot(data=df, orient='h', palette='Set3')
    else:
        plt.figure(figsize=(20, 20))
        sns.boxplot(data=df[column], orient='h', palette='Set3')
    plt.xlabel("Valor")
    plt.show()

def heat_map(df):
    sns.set(style="whitegrid", font_scale=1)
    plt.figure(figsize=(15, 15))
    sns.heatmap(df.corr(), vmax=0.8, square=True, cmap="GnBu", linecolor="r", annot=True, annot_kws={'size': 10})
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
    Counts the number of missing (null) values in each column of a DataFrame.
    
    Parameters:
    - df (pd.DataFrame): The DataFrame containing the data.
    
    Returns:
    - pd.Series: A series with the column names and the number of missing values in each column.
    """
    null_counts = df.isnull().sum()
    return null_counts


def count_unique_values(df):
    """
    Returns a Series with the number of unique values for each column in the DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.Series: A Series where the index is the column names and the values are the number of unique values.
    """
    unique_counts = df.nunique()
    return unique_counts
