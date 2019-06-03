import os
from typing import List
from tqdm import tqdm

import requests
import numpy as np
from PIL import Image
import json

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
        # assert isinstance(image[0][0][0], int)
        json_data =  {"instances": [image]}
        json_str = json.dumps(json_data)
        # json_str = r'{"instances": 12}'
        # print(json_str)
        # print(json.loads(json_str))
        response = requests.post(url, data=json_str)
        print(response)
        return response.json()

    def classify_images(self, paths: List):
        results = []
        for path in tqdm(paths):
            image = load_image(path)
            results.append([self.classify_image(image)])
        return results


def load_image(path: str):
    im = Image.open(path)
    reshaped = im.resize((256, 256))  # must match serving reciever fn config
    matrix = np.array(reshaped).astype(float) # .astype(np.uint8)
    assert matrix.shape == (256, 256, 3)
    return matrix.tolist()

if __name__ == '__main__':

    host = '3.82.176.136'
    port = '8501'
    model_name = '1559567967'
    hard_image_loc = 'zoobot/tests/test_examples/example_a.png'  # relative path, careful
    assert os.path.exists(hard_image_loc)
    smooth_image_loc = 'zoobot/tests/test_examples/smooth.png'  # relative path, careful
    assert os.path.exists(smooth_image_loc)
    featured_image_loc = 'zoobot/tests/test_examples/featured.png'

    # image = load_image(featured_image_loc)
    # with open('temp.txt', 'w') as f:
    #     f.write(str(image))
    # exit(0)

    server = Server(host, port, model_name)
    status = server.check_up()
    print(status)

    # image = load_image(hard_image_loc)
    # print(server.classify_image(image))


    results = server.classify_images([hard_image_loc, smooth_image_loc, featured_image_loc])
    print(results)