import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
 
# 1. Load Data & Identify Features
df = sns.load_dataset("iris")
numeric_cols = df.select_dtypes(include="number").columns.tolist()
print(f"Features:\n{df.dtypes}\n\nNumeric: {numeric_cols}")
 
# 2. Histograms (Distributions)
df.hist(figsize=(10, 6), color='skyblue', edgecolor='black')
plt.suptitle("Feature Histograms")
plt.tight_layout()
plt.show()
 
# 3. Boxplots (Identify Outliers)
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols], palette="Set2")
plt.title("Boxplots Before Outlier Handling")
plt.show()
 
# 4. Handle Outliers using IQR Capping (Winsorization)
for col in numeric_cols:
    Q1, Q3 = df[col].quantile(0.25), df[col].quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5 * IQR, Q3 + 1.5 * IQR
 
    # Identify and Cap
    outliers = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"{col}: Capped {outliers} outliers")
 
# 5. Verify Results
plt.figure(figsize=(10, 6))
sns.boxplot(data=df[numeric_cols], palette="Pastel1")
plt.title("Boxplots After Outlier Handling")
plt.show()