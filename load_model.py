import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import load_model

import autokeras as ak

(x_train, y_train), (x_test, y_test) = mnist.load_data()

loaded_model = load_model('model_autokeras', custom_objects=ak.CUSTOM_OBJECTS)

predicted_y = loaded_model.predict(tf.expand_dims(x_test, -1))
print(predicted_y)


