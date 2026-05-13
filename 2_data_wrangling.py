import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
# Creating a dictionary with intentional issues
data = {
    'Student_ID': range(1, 51),
    'Gender': ['M', 'F', 'f', 'M', 'm', 'F'] * 8 + ['M', 'F'], # Inconsistent casing
    'Math_Score': [65, 70, 75, np.nan, 80, 150, 90, 20, 85, 72] * 5, # Outlier 150, Missing values
    'Reading_Score': [88, 92, 95, 70, np.nan, 85, 78, 10, 90, 82] * 5, # Outlier 10
    'Writing_Score': np.random.randint(60, 100, 50),
    'Placement_Score': [75, 80, 85, 90, 95, 20, 30, 40, 50, 60] * 5
}
df = pd.DataFrame(data)
print("Initial Dataset Info:")
df.info()
 
df['Gender'] = df['Gender'].str.upper() # Standardize to 'M' and 'F'
 
df['Math_Score'] = df['Math_Score'].fillna(df['Math_Score'].mean())
df['Reading_Score'] = df['Reading_Score'].fillna(df['Reading_Score'].median())
print("Missing values after handling:")
print(df.isnull().sum())
 
sns.boxplot(df['Math_Score'])
 
# Handling Outliers using IQR
Q1 = df['Math_Score'].quantile(0.25)
Q3 = df['Math_Score'].quantile(0.75)
IQR = Q3 - Q1
lower_limit = Q1 - 1.5 * IQR
upper_limit = Q3 + 1.5 * IQR
 
df['Math_Score'] = np.where(df['Math_Score'] > upper_limit, upper_limit, 
                   np.where(df['Math_Score'] < lower_limit, lower_limit, df['Math_Score']))
print("Outliers handled using Capping method.")
sns.boxplot(df['Math_Score'])
 
# Check Skewness and log transform
print("Original Skewness", df['Placement_Score'].skew())
sns.histplot(df['Placement_Score'], kde=True)
 
df['Placement_Log'] = np.log1p(df['Placement_Score'])
print("After Log Transformation skewness", df['Placement_Score'].skew())
sns.histplot(df['Placement_Score'], kde=True)