import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train / 255.0
x_test = x_test / 255.0
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

model = tf.keras.models.load_model("model.h5")

tensorboard = tf.keras.callbacks.TensorBoard(log_dir="tensorboard")


# tf.keras.utils.plot_model(model, to_file="model.png")
model.fit(x_train, y_train, initial_epoch=21, batch_size=128, epochs=40, callbacks=[tensorboard])
model.save("model.h5")











