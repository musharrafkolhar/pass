import seaborn as sns
import matplotlib.pyplot as plt
 
# Load the inbuilt titanic dataset
titanic = sns.load_dataset('titanic')
 
# Display dataset
titanic.head()
 
# Create Histogram
plt.figure(figsize=(10, 6))
sns.histplot(titanic['fare'], bins=40, kde=True, color='blue')
plt.title('Distribution of Passenger Fares')
plt.xlabel('Fare')
plt.ylabel('Frequency')
plt.show()
 
# Bar plot to see survival rate across Gender
plt.figure(figsize=(8, 5))
sns.barplot(x='sex', y='survived', data=titanic)
plt.title('Survival Rate by Gender')
plt.show()
 
# Boxplot to see the distribution of fares across different classes
plt.figure(figsize=(10, 6))
sns.boxplot(x='class', y='fare', data=titanic, palette='rainbow')
plt.title('Fare Distribution by Passenger Class')
plt.show()