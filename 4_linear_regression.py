import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error ,r2_score
 
df=pd.read_csv('HousingData.csv')
df.head()
 
df.info()
 
df.isnull().sum()
 
X=df.drop("MEDV",axis=1)
y=df["MEDV"]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
 
y_pred=model.predict(X_test)
 
mse=mean_squared_error(y_test,y_pred)
mae=mean_absolute_error(y_test,y_pred)
print("Mean Squared Error:", mse)
print("Mean Absolute Error:",mae)
 
r2=r2_score(y_test,y_pred)
print("R2 Score:", r2)
 
plt.scatter(y_test,y_pred)
plt.xlabel("Actual Prices")
plt.ylabel("Predicted Prices")
plt.title("Actual vs Predicted House Prices")
plt.show()