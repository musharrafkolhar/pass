import tensorflow as tf

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# normalizing pixel values
X_train = X_train / 255.0
X_test = X_test / 255.0

# model , cnn box , color channel for grey is 1 therefore 28x28x1 ,
model = tf.keras.Sequential([
    tf.keras.layers.Reshape((28, 28, 1), input_shape=(28, 28)),
    tf.keras.layers.Conv2D(32, (3, 3), activation="relu"),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dense(10, activation="softmax")
])

# compile
model.compile(optimizer="adam",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])

# train
model.fit(X_train, y_train, epochs=3)

# evaluate
model_loss, model_acc = model.evaluate(X_test, y_test)
print(f"Final accuracy : {model_acc*100:.2f}%")