import pandas as pd


def missing_value_summary(df):
    """
    Returns a DataFrame summarizing the number and percentage of missing values in each column of the input DataFrame.

    Parameters:
    df (pd.DataFrame): The input DataFrame.

    Returns:
    pd.DataFrame: A DataFrame with two columns:
        - 'Missing Values': Number of missing values per column
        - 'Missing Percentage': Percentage of missing values per column
    """
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100

    summary_df = pd.DataFrame({
        'Missing Values': missing_values,
        'Missing Percentage': missing_percentage.round(2)
    })

    return summary_df[summary_df['Missing Values'] > 0]