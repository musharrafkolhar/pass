import tensorflow as tf
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# prepare data
X, y = make_classification(n_samples=1000, n_features=10, random_state=42)

# train test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# ==========================================
# Logistic regression , 0 hidden layers
# ==========================================
log_reg_model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, activation="relu", input_shape=(10,))
])

log_reg_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train it
log_reg_model.fit(X_train, y_train, epochs=10, verbose=1)

# evaluating on the unseen data
log_loss, log_acc = log_reg_model.evaluate(X_test, y_test)
print(f"Final accuracy: {log_acc*100:.2f}%")


# ==========================================
# Neural Network
# ==========================================
nn_model = tf.keras.Sequential([
    tf.keras.layers.Dense(16, activation="relu", input_shape=(10,)),
    tf.keras.layers.Dense(8, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# compile the model
nn_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

# train it
nn_model.fit(X_train, y_train, epochs=10, verbose=1)

# evaluating on unseen data
nn_loss, nn_acc = nn_model.evaluate(X_test, y_test)

print(f"Final accuracy {nn_acc*100:.2f}%")