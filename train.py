import os
import sys
import numpy as np
from random import shuffle
from gan import Gan
from support.image_caption_loader import load_image_and_text

def main():
    img_width = 64
    img_height = 64
    img_channels = 3

    image_path = './dataset/images'
    caption_path = './dataset/captions'
    weights_path = './weights'

    image_caption_pair = load_image_and_text(img_width , img_height , image_path , caption_path)
    shuffle(image_caption_pair)

    gan = Gan()
    gan.img_width = img_width
    gan.img_height = img_height
    gan.img_channels = img_channels
    gan.random_input_dim = 100
    gan.glove_path = './glove'

    batch_size = 20
    epoch = 1

    gan.fit(image_caption_pair , epoch , batch_size , './output/intermediate_output' , 50 , weights_path)

if __name__ == '__main__':
    main()
