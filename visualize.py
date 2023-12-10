import pandas as pd
import matplotlib.pyplot as plt

def compute_correlation_and_plot_all(df, column2):
    columns_to_exclude = [column2]
    for column in df.columns:
        if column not in columns_to_exclude:
            df_subset = df[[column, column2]].dropna()
            correlation = df_subset[column].corr(df_subset[column2])
            
            plt.figure(figsize=(8, 6))
            plt.scatter(df_subset[column], df_subset[column2], alpha=0.5)
            plt.title(f'{column} vs {column2}')
            plt.xlabel(column)
            plt.ylabel(column2)
            plt.grid(True)
            plt.show()
            
            print(f'Correlation between {column} and {column2}: {correlation}\n')