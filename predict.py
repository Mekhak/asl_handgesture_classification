from PIL import Image
import numpy as np
import joblib
import argparse
from pathlib import Path
import pickle

def parse_args():
    '''
    parse arguments
    :return: arguments
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path)
    args = parser.parse_args()

    return args

def predict(args):
    '''
    :param args: arguments
    :return: prediction result
    '''

    model_path = r'models\logistic.sav'
    loaded_model = joblib.load(model_path)
    width = height = 150
    img = np.asarray(Image.open(args.image).resize((width, height))).flatten()

    target_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H',
                    'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z']

    index = loaded_model.predict([img])[0]

    return target_names[index]


if __name__ == '__main__':
    args = parse_args()
    print("result: ", predict(args))

