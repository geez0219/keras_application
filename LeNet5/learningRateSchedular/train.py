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

model.compile(optimizer=tf.keras.optimizers.Adam(lr=0),
              loss=tf.keras.losses.categorical_crossentropy,
              metrics=[tf.keras.metrics.categorical_accuracy])


def learning_function(epoch):
    if epoch < 10:
        return 0.0001
    else:
        return 0.0


# define new Tensorboard to add new metrics
class XTensorboard(tf.keras.callbacks.TensorBoard):
    def on_epoch_end(self, epoch, logs=None):
        logs.update({"lr": tf.keras.backend.eval(self.model.optimizer.lr)})
        super().on_epoch_end(epoch, logs)


tb_cb = XTensorboard(log_dir="tensorboard")
# this callback function can affect learning rate of optimizer (yes, even though it doesn't know which optimizer it has)
ls_cb = tf.keras.callbacks.LearningRateScheduler(learning_function, verbose=1)
model.fit(x_train, y_train, batch_size=128, epochs=20, callbacks=[ls_cb, tb_cb])











