# UTILITY FUNCTION - Add this to a utils file or at the top of your modules
# This will help prevent DataFrame/Series issues throughout your codebase

import pandas as pd
from typing import Union

def safe_get_series(df: pd.DataFrame, column: Union[str, list]) -> pd.Series:
    """
    Safely extract a Series from a DataFrame.
    
    Handles the case where df[column] might return a DataFrame instead of a Series.
    
    Args:
        df: DataFrame to extract from
        column: Column name (string) or list with single column name
    
    Returns:
        Series extracted from the DataFrame
    
    Examples:
        >>> series = safe_get_series(df, 'language')
        >>> series = safe_get_series(df, ['language'])  # Also works
    """
    # Handle list input
    if isinstance(column, list):
        if len(column) == 0:
            raise ValueError("Column list is empty")
        column = column[0]
    
    # Extract the column
    result = df[column]
    
    # If it's a DataFrame, extract the first (and should be only) column
    if isinstance(result, pd.DataFrame):
        result = result.iloc[:, 0]
    
    return result


# EXAMPLE USAGE:

# Instead of:
# unique_langs = df['language'].unique()

# Use:
# lang_series = safe_get_series(df, 'language')
# unique_langs = lang_series.unique()

# Or more concisely:
# unique_langs = safe_get_series(df, 'language').unique()


# For value_counts:
# Instead of:
# counts = df['language'].value_counts()

# Use:
# counts = safe_get_series(df, 'language').value_counts()


# This pattern works for any Series method:
# - .unique()
# - .value_counts()
# - .drop_duplicates()
# - .mean(), .std(), etc.
# - Any other Series-specific method