import pandas as pd
import numpy as np
 
# 1. Create a sample dataset
data = {
    'Age_Group': ['Young', 'Middle-Aged', 'Senior', 'Young', 'Middle-Aged', 'Senior', 'Young', 'Middle-Aged'],
    'Income': [45000, 75000, 60000, 48000, 80000, 62000, 50000, 82000]
}
df_income = pd.DataFrame(data)
 
# 2. Group by Age_Group and calculate summary statistics for Income
summary_stats = df_income.groupby('Age_Group')['Income'].agg(['mean', 'median', 'min', 'max', 'std'])
print("Summary Statistics of Income Grouped by Age Group:")
display(summary_stats)
 
# 3. Create a list that contains a numeric value for each response to the categorical variable
# Mapping: Young=1, Middle-Aged=2, Senior=3
mapping = {'Young': 1, 'Middle-Aged': 2, 'Senior': 3}
numeric_responses = df_income['Age_Group'].map(mapping).tolist()
 
print("\nNumeric list for Categorical responses (Age_Group):")
print(numeric_responses)
import seaborn as sns
 
# Load the Iris dataset from seaborn's built-in library
iris = sns.load_dataset('iris')
 
# Filter data for specific species # we compare all three: setosa, versicolor, and virginica.
species_list = ['setosa', 'versicolor', 'virginica']
 
for species in species_list:
    print(f"\n--- Statistical Details for Species: {species} ---")
 
    # Filter the dataframe for the current species
    species_data = iris[iris['species'] == species]
 
    # Use describe() to get mean, std, percentiles, min, and max
    # We transpose () it for better readability in Jupyter
    display(species_data.describe().T)