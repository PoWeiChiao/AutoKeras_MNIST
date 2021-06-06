import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import autokeras as ak

print(tf.__version__)

(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape)
print(y_train.shape)
print(y_train[:3])

clf = ak.ImageClassifier(overwrite=True, max_trials=1)
clf.fit(x_train, y_train, epochs=10)

predicted_y = clf.predict(x_test)
print(predicted_y)

print(clf.evaluate(x_test, y_test))

model = clf.export_model()
print(type(model))

try:
    model.save('model_autokeras', save_format='tf')
except Exception:
    model.save('model_autokeras.h5')


