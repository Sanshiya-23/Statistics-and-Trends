# -*- coding: utf-8 -*-
"""
Created on Wed May 17 15:10:40 2023

@author: SANSHIYA
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
import seaborn as sns


def dataframe_maker(url):
    """
    The function reads data from a CSV file at the provided URL and filters it
    based on desired indicators, countries, and years, resulting in two cleaned
    dataframes: df_years and df_countries.

    Args:
        - url (str): The URL of the CSV file to read.

    Returns:
        - df_years (pandas.DataFrame): A dataframe containing the filtered and
        cleaned data with selected years, countries and indicators.
        - df_countries (pandas.DataFrame): A dataframe containing the
        transposed and processed data with countries as columns.
    """

    # Read data from CSV file
    df = pd.read_csv(url, skiprows=4)

    # Define the list of BRICS countries
    brics = ['Brazil', 'Russia', 'India', 'China', 'South Africa']

    # Define the list of desired indicators
    desired_indicators = [
        'CO2 emissions (kt)',
        'Renewable energy consumption (% of total final energy consumption)',
        'Forest area (% of land area)',
        'Population growth (annual %)'
    ]

    # Filter the original dataframe by countries list and desired indicators
    df_brics = df[(df['Country Name'].isin(brics)) & (
        df['Indicator Name'].isin(desired_indicators))]

    # Select the desired years and set them as columns
    years = ['2000', '2005', '2010', '2015', '2018']
    df_years = df_brics.loc[:, [
        'Country Name', 'Country Code', 'Indicator Name'] + years]

    # Remove rows with missing values
    df_years.dropna(inplace=True)

    # Reset the index
    df_years = df_years.reset_index(drop=True)

    # Transpose DataFrame without index
    df_countries = df_years.set_index('Country Name').transpose()

    return df_years, df_countries


# Dataset file
url = r"C:\Users\SANSHIYA\Downloads\Climate Change Dataset.csv"

# Call the function with the URL argument
df_years, df_countries = dataframe_maker(url)

# Print the two resulting dataframes
print(df_years)
print(df_countries)

# Save two DataFrames as CSV files
df_years.to_csv('df_years.csv')
df_countries.to_csv('df_countries.csv')

# Compute summary statistics for each indicator and country
stats = df_years.groupby(['Indicator Name', 'Country Name']).describe()

# Print the summary statistics for each indicator and country
print(stats)

# Group the data by Indicator Name
grouped_data = df_years.groupby(['Indicator Name'])

# Create a bar chart for each indicator
for name, group in grouped_data:
    # Set up the plot
    fig, ax = plt.subplots()
    # Add a title to the plot
    plt.title(name)
    ax.set_xticklabels(df_years['Country Name'].unique())
    ax.set_xticks(np.arange(len(df_years['Country Name'].unique())))
    ax.set_xlabel('Country Name')
    ax.set_ylabel(name)

    # Plot the data
    for i, year in enumerate(['2000', '2005', '2010', '2015', '2018']):
        ax.bar(np.arange(len(group['Country Name'].unique())) + i*0.15,
               group[year].values,
               width=0.15,
               label=year)

    # Add a legend
    ax.legend()

    # Show the plot
    plt.show()

# Create a pivot table for heatmap visualization
df_pivot = df_years.pivot_table(values=['2015'],
                                index=['Country Name', 'Country Code'],
                                columns='Indicator Name')
df_pivot.columns = df_pivot.columns.droplevel(0)
corr_matrix = df_pivot.corr()

# Plot the heatmap with updated colors
plt.figure(figsize=(10, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')

# Show the plot
plt.show()
