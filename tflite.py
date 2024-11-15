import tensorflow as tf

model = tf.keras.models.load_model("final_models/voice_verification_model.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
tflite_model = converter.convert()

with open("final_models/voice_verification_model.tflite", "wb") as f:
    f.write(tflite_model)
