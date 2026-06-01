import tensorflow as tf
import tensorflow_hub as hub

MODEL_PATH = "models/mobilenet_v2.tflite"

model = tf.keras.applications.MobileNetV2(
    weights="imagenet"
)

converter = tf.lite.TFLiteConverter.from_keras_model(model)

tflite_model = converter.convert()

with open(MODEL_PATH, "wb") as f:
    f.write(tflite_model)

print(f"Saved TFLite model to {MODEL_PATH}")