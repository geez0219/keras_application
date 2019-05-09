import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)


model = tf.keras.Sequential([
    tf.keras.layers.Conv2D(filters=6, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu, input_shape=[32,32,3]),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2], padding="valid"),
    tf.keras.layers.Conv2D(filters=16, kernel_size=[5, 5], padding="valid", activation=tf.nn.relu),
    tf.keras.layers.MaxPooling2D(pool_size=[2, 2], padding="valid"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(120, activation=tf.nn.relu),
    tf.keras.layers.Dense(84, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tensorboard")

# tf.keras.utils.plot_model(model, to_file="model.png")
model.fit(x_train, y_train, batch_size=128, epochs=20, callbacks=[tensorboard])
model.save("model.h5")  # use model save also save model architecture, weigth and optimizer











