import os

from protopost import ProtoPost
from nd_to_json import nd_to_json
from img_to_b64 import b64_to_img

from keras_apps import KerasApp

PORT = int(os.getenv("PORT", 80))
APP_NAME = os.getenv("APP_NAME", "mobilenet_v2")
POOLING = os.getenv("POOLING", "max")
POOLING = None if POOLING.lower() == "none" else POOLING
RESIZE_TO = os.getenv("RESIZE_TO", None)

#TODO: predownload model...

#https://keras.io/api/applications/mobilenet/
#looks like preprocess_inputs assume the image is 0..255 range
#https://www.tensorflow.org/api_docs/python/tf/keras/applications/mobilenet/preprocess_input

app = KerasApp(app_name=APP_NAME, pooling=POOLING, resize_to=RESIZE_TO)

def extract_features(data):
  #The preprocessed data are written over the input data if the data types are compatible. To avoid this behaviour, numpy.copy(x) can be used.
  #NOTE ^ this, if we plan to re-use img
  img = b64_to_img(data)
  # img = img / 255.#??
  features = app.run(img)
  return nd_to_json(features)

routes = {
  "": extract_features
}

ProtoPost(routes).start(PORT)
