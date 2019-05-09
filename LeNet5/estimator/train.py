import tensorflow as tf
import numpy as np

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train = x_train/255.0
x_test = x_test/255.0
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

# define model in tensor style
# but cannot blend with tensorflow tensor
input = tf.keras.layers.Input([32,32,3])  # the input is tensor (placeholder)
x = tf.keras.layers.Conv2D(filters=6, kernel_size=[5,5], activation=tf.nn.relu)(input)
x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
x = tf.keras.layers.Conv2D(filters=16, kernel_size=[5,5], activation=tf.nn.relu)(x)
x = tf.keras.layers.MaxPooling2D(strides=[2,2])(x)
x = tf.keras.layers.Flatten()(x)
x = tf.keras.layers.Dense(units=84, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=120, activation=tf.nn.relu)(x)
x = tf.keras.layers.Dense(units=10, activation=tf.nn.softmax)(x)
model = tf.keras.models.Model(inputs=input, outputs=x)


model.compile(loss=tf.keras.losses.categorical_crossentropy,
              optimizer=tf.keras.optimizers.Adam(),
              metrics=[tf.keras.metrics.categorical_accuracy])

est_model = tf.keras.estimator.model_to_estimator(keras_model=model)

train_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1":x_train},
    y=y_train,
    shuffle=True,
    num_epochs=None
)

eval_input_fn = tf.estimator.inputs.numpy_input_fn(
    x={"input_1":x_test},
    y=y_test,
    shuffle=True
)

est_model.train(input_fn=train_input_fn, steps=100*int(50000/128))
a = est_model.evaluate(input_fn=eval_input_fn, steps=100)
print(a)





