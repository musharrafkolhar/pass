import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
 
# Load dataset
df = pd.read_csv('Social_Network_Ads.csv')
 
df['Gender'] = df['Gender'].map({'Male': 0, 'Female': 1})
 
# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
 
# Train model
model = LogisticRegression()
model.fit(X_train, y_train)
print("Logistic Regression model trained")
 
# Predict
y_pred = model.predict(X_test)
 
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix generated")
 
# Extract values
tn, fp, fn, tp = cm.ravel()
 
# Metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
error_rate = 1 - accuracy
precision = tp / (tp + fp)
recall = tp / (tp + fn)
print('TP:', tp)
print('FP:', fp)
print('TN:', tn)
print('FN:', fn)
print('Accuracy:', accuracy)
print('Error Rate:', error_rate)
print('Precision:', precision)
print('Recall:', recall)