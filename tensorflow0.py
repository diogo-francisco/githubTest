import tensorflow
tensorflow.compat.v1.logging.set_verbosity(tensorflow.compat.v1.logging.ERROR)

import numpy

celsius_q = numpy.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
fahrenheit_a = numpy.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

for i, c in enumerate(celsius_q):
	print("{} degrees Celsius = {} degrees Fahrenhet".format(c, fahrenheit_a[i]))

#create layer (I think)
layer0 = tensorflow.keras.layers.Dense(units=1, input_shape=[1])

#create model and give it the layer
model = tensorflow.keras.Sequential([layer0])

#configures the model for training
#configs LOSSes and metrics
#(the higher the Adam's value, the more precise it is)
model.compile(optimizer=tensorflow.keras.optimizers.Adam(learning_rate = 0.1), 
	loss='mean_squared_error')

#training proccess using an optimization proccess -> Gradient Descent
history = model.fit(celsius_q, fahrenheit_a, epochs=1000, verbose=1)

#predict result
model.predict([100.0])

#output predicted result
print(model.predict([100.0]))
