import numpy as np
import tensorflow as tf 
import argparse
import googlenet
import alexnet

def google():
    return (googlenet.googlenet(), "googlenet")

def alex():
    return (alexnet.alexnet(), "alexnet")
    
def vgg16():
    return (tf.keras.applications.vgg16.VGG16(
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        input_shape=None,
        pooling=None,
        classes=1000,
    ), "vgg16")
    
def mobilenetV2():
    return (tf.keras.applications.mobilenet_v2.MobileNetV2(
        input_shape=(224, 224, 3),
        alpha=1.0,
        include_top=True,
        weights='imagenet',
        input_tensor=None,
        pooling=None,
        classes=1000,
        classifier_activation='softmax',
    ), "mobilenetv2")


def representative_dataset_gen():
    for _ in range(1):
      data = np.random.rand(1, 224, 224, 3)
      yield [data.astype(np.float32)]

models = [mobilenetV2, vgg16, alex, google]

for m_func in models:
    model, name = m_func()

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()
    with open(name + "-.tflite", 'wb') as f:
        f.write(tflite_model)

    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    # converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # converter.representative_dataset = representative_dataset_gen
    # converter.inference_input_type = tf.uint8
    # converter.inference_output_type = tf.uint8
    # tflite_model = converter.convert()

    # # Save the model.
    # with open(name + "-quant.tflite", 'wb') as f:
    #     f.write(tflite_model)

