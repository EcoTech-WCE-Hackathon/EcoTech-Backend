import pickle
import joblib
import random
import os
import numpy as np
import tensorflow as tf

print(os.path.dirname(__file__))
print(os.path.join(os.path.dirname(__file__), "../static/model.pkl"))


class Predictor:
    img_height = 180
    img_width = 180
    class_names = ["ewaste", "not_ewaste"]
    model_loaded = joblib.load(os.path.abspath("../server/static/model.pkl"))
    model = pickle.loads(model_loaded)

    @staticmethod
    def __init__():
        # train model
        pass

    @staticmethod
    def predict(image):
        val = random.uniform(0, 1)
        print(val)
        return val > 0.5

    @staticmethod
    def getPrediction(image=""):
        img = tf.keras.utils.load_img(
            f"{os.path.abspath('media/images/')}/rohinmouseagain.jpg",
            target_size=(Predictor.img_height, Predictor.img_width),
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create a batch
        predictions = Predictor.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        print(100 * np.max(score))
        print(np.argmax(score))
        print(score)
        if np.argmax(score) == 1:
            return 0
        else:
            return 1
