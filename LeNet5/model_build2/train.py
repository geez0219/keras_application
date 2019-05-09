import tensorflow as tf

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# model build using a sequential
model = tf.keras.models.Sequential()

# the first layer can optionally specify the input_shape.
model.add(tf.keras.layers.Conv2D(filters=6, kernel_size=[5,5], activation=tf.nn.relu, input_shape=[32,32,3]))
model.add(tf.keras.layers.MaxPooling2D(strides=[2,2]))
model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=[5,5], activation=tf.nn.relu))
model.add(tf.keras.layers.MaxPooling2D(strides=[2,2]))
model.add(tf.keras.layers.Flatten())
model.add(tf.keras.layers.Dense(units=84, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=120, activation=tf.nn.relu))
model.add(tf.keras.layers.Dense(units=10, activation=tf.nn.softmax))

model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.categorical_accuracy])

tb_cb = tf.keras.callbacks.TensorBoard(log_dir="tensorboard")
cp_cb = tf.keras.callbacks.ModelCheckpoint(filepath="checkpoint.h5")

# model.fit(x_train, y_train, epochs=10, validation_split=0.1, callbacks=[tb_cb, cp_cb])
model.summary()


