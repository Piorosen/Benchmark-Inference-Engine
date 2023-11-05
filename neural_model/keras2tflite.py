#%%
import tensorflow as tf
#%%
m = tf.keras.applications.mobilenet_v2.MobileNetV2(input_shape=(224,224,3), weights='imagenet',classes=1000)
#%%
model_name = "mobilenet_v2"
converter = tf.lite.TFLiteConverter.from_keras_model(m)
tflite_model = converter.convert()
#%%
# Save the model.
with open(model_name + ".tflite", 'wb') as f:
    f.write(tflite_model)
# %%
