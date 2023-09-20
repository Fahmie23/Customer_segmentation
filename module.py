#%%
# Importing libraries
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#%%
def missing_values(dataframe):
    """
    Print the count and percentage of missing values in each column of a Pandas DataFrame.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame to check for missing values.
    """
    # Calculate the total number of missing values in each column
    missing_values = dataframe.isnull().sum()
    
    # Calculate the percentage of missing values in each column
    total_rows = dataframe.shape[0]
    missing_percentage = (missing_values / total_rows) * 100
    
    # Create a DataFrame to display the results
    missing_info = pd.DataFrame({
        'Column': missing_values.index,
        'Missing Values': missing_values.values,
        'Percentage': missing_percentage.values
    })
    
    # Print the missing values information
    print("Missing Values Information:")
    print(missing_info)

#%%
def drop_columns(dataframe, columns_to_drop):
    """
    Drop specified columns from a Pandas DataFrame.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame from which to drop columns.
    - columns_to_drop (list): A list of column names to be dropped.
    
    Returns:
    - pd.DataFrame: A new DataFrame with the specified columns removed.
    """
    new_dataframe = dataframe.drop(columns=columns_to_drop)
    return new_dataframe

#%%
def fill_missing_values(dataframe, fill_values_dict):
    """
    Fill missing (NaN) values in a Pandas DataFrame with a specified value.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame containing missing values to be filled.
    - fill_value: The value to use for filling missing values.
    
    Returns:
    - pd.DataFrame: A new DataFrame with missing values filled.
    """
    filled_dataframe = dataframe.fillna(fill_values_dict)
    return filled_dataframe

#%%
def remove_rows(dataframe, columns_to_consider):
    """
    Remove rows with missing (NaN) values for specific columns in a Pandas DataFrame.
    
    Parameters:
    - dataframe (pd.DataFrame): The DataFrame from which to remove rows with missing values.
    - columns_to_consider (list): A list of column names to consider when removing rows.
    
    Returns:
    - pd.DataFrame: A new DataFrame with rows containing missing values for the specified columns removed.
    """
    cleaned_dataframe = dataframe.dropna(subset=columns_to_consider)
    return cleaned_dataframe
