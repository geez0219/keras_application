import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


model = tf.keras.models.load_model("checkpoint.h5")
# model.metric_names

loss, accuracy = model.evaluate(x_test, y_test)

print("loss:{}, acc:{}".format(loss, accuracy))

predicted = model.predict(x_test)
