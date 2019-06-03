import os
from Typing import List

import requests
import numpy as np
from PIL import Image

class Server:

    def __init__(self, host: str, port: str, model_name: str):
        self.host = host
        self.port = port
        self.model_name = model_name

    def check_up(self):
        url = 'http://{}:{}/v1/models/{}'.format(self.host, self.port, self.model_name)  # all models will be checked
        response = requests.get(url)
        print(response)
        return response.json()

    def classify_image(self, image: List):
        url = 'http://{}:{}/v1/models/{}:predict'.format(self.host, self.port, self.model_name)
        assert isinstance(image, list)
        assert isinstance(image[0][0][0], np.uint8)
        data =  {
            "instances": image
        }
        response = requests.post(url, data)
        print(response)
        return response.json


def load_image(path: str):
    im = Image.open(path)
    reshaped = im.resize((256, 256))  # must match serving reciever fn config
    matrix = np.array(reshaped).astype(np.uint8)
    assert matrix.shape == (256, 256, 3)
    return matrix

if __name__ == '__main__':

    host = '3.82.176.136'
    port = '8501'
    model_name = 'TODO'
    test_image_loc = 'zoobot/tests/test_examples/example_a.png'  # relative path, careful
    assert os.path.exists(test_image_loc)

    server = Server(host, port, model_name)
    server.check_up()
    image = load_image(test_image_loc)
    server.classify_image(image)