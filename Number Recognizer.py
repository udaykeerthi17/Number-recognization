import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = mnist.load_data()

x_train = x_train / 255.0
x_test = x_test / 255.0

model = Sequential([
    Flatten(input_shape=(28, 28)),
    Dense(512, activation='relu'),
    Dense(512, activation='relu'),
    Dense(10, activation='softmax')
])

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test accuracy: {test_acc}")

def predict_digit(image):
    image = tf.convert_to_tensor(image)
    image = tf.expand_dims(image, axis=0)
    prediction = model.predict(image)
    digit = tf.argmax(prediction, axis=1).numpy()[0]
    return digit

example_index = 0
predicted_digit = predict_digit(x_test[example_index])

plt.imshow(x_test[example_index], cmap='gray')
plt.title(f"Predicted Digit: {predicted_digit}")
plt.show()
