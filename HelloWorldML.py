import tensorflow as tf

from tensorflow import keras

model = tf.keras.Sequential([keras.layers.Dense(units=1,input_shape=[1])])
model.compile(optimizer='sgd',loss='mean_squared_error')
xs=[1.0,2.0,3.0,4.0,5.0,6.0]
ys=[2.0,3.0,4.0,5.0,6.0,7.0]

model.fit(xs,ys,epochs=5000)
print(model.predict([1.0]))
