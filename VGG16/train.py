import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)


def VGG16():
    input_tensor = tf.keras.layers.Input(shape=(32,32,3))
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(input_tensor)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=64, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=128, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=256, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Conv2D(filters=512, kernel_size=[3,3], padding="same",
                               activation=tf.nn.relu)(x)
    x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=4096, activation=tf.nn.relu, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax, kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

    return tf.keras.models.Model(inputs=input_tensor, outputs=x)


model = VGG16()
model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.categorical_accuracy])

tb_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard")
cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint.h5")

model.fit(x_train, y_train, batch_size=128, epochs=100, validation_split=0.1, callbacks=[tb_cb, cp_cb])




