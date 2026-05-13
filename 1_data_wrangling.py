import pandas as pd
import numpy as np
pd.set_option('display.max_columns', None)
# Load the dataset directly from a URL (Open Source: GitHub/Kaggle source)
url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
df = pd.read_csv(url)
print("Dataset loaded successfully!")
df.head()
 
print(f"Shape of the dataset: {df.shape}\n") # 1. Check dimensions
print("Missing values per column:") # 2. Identify missing values
print(df.isnull().sum())
print("\nStatistical Summary (Numeric):") # 3. Initial Statistics
df.describe()
 
print("Data Types of Variables:")
print(df.dtypes)
 
df['Survived'] = df['Survived'].astype('category')
df['Pclass'] = df['Pclass'].astype('category')
 
df['Fare_Normalized'] = (df['Fare'] - df['Fare'].min()) / (df['Fare'].max() - df['Fare'].min())
print("Updated Data Types:")
print(df.dtypes[['Survived', 'Pclass', 'Fare_Normalized']])
df[['Fare', 'Fare_Normalized']].head()
 
# Binary Encoding for 'Sex'
df['Sex'] = df['Sex'].map({'male': 1, 'female': 0})
# 2. One-Hot Encoding for 'Embarked'
df = pd.get_dummies(df, columns=['Embarked'], prefix='Port')
print("Transformation Complete. New columns for 'Embarked' created:")
df.head()