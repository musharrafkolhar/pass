import seaborn as sns
import matplotlib.pyplot as plt
df = sns.load_dataset('titanic')
sns.boxplot(x='sex', y='age', hue='survived', data=df)
plt.title('Age Distribution by Gender and Survival')
plt.xlabel('Gender')
plt.ylabel('Age')
plt.show()
