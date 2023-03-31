import numpy as np
import tensorflow as tf 
import argparse
import googlenet

def google():
    return googlenet.googlenet()
    
def vgg16():
    return tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    )
    
def mobilenetV2():
    return tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
    )


def representative_dataset_gen():
    for _ in range(1):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]

# model = resnet101()
model = google()
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_gen
converter.inference_input_type = tf.uint8
converter.inference_output_type = tf.uint8
tflite_model = converter.convert()

# Save the model.
with open("google-quant.tflite", 'wb') as f:
    f.write(tflite_model)

model = vgg16()
# model = mobilenetV2()
