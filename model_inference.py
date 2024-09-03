import joblib
import numpy as np
import tensorflow as tf

x = np.array(joblib.load('mof_vectors.pkl'))
x = x.reshape(x.shape[0], x.shape[1], x.shape[3])

model = tf.keras.models.load_model('ads_cap_models/ads_cap_model_msle_best.keras')

print(model(tf.expand_dims(x[3647], axis=0)))