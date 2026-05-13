import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score
 
# Load dataset
df = pd.read_csv('iris.csv')
# Features and target
X = df.iloc[:, :-1]
y = df.iloc[:, -1]
 
# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
X, y, test_size=0.2, random_state=42
)
 
# Train model
model = GaussianNB()
model.fit(X_train, y_train)
print("Naive Bayes model trained")
 
# Predict
y_pred = model.predict(X_test)
# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion matrix displayed")
print('Confusion Matrix:', cm)
 
# Metrics
accuracy = accuracy_score(y_test, y_pred)
error_rate = 1 - accuracy
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
print('Accuracy:', accuracy)
print('Error Rate:', error_rate)
print('Precision:', precision)
print('Recall:', recall)