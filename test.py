import os
import sys
import numpy as np
from random import shuffle
from gan import Gan

def main():

    gan=Gan()
    gan.load_model()

    for i in range(10):
        ip = input('Enter image description\n-')
        for j in range(10):
            gen_img = gan.generate_image_from_text(ip)
            gen_img.save('./output/' + 'result' + str(i) + '-' + str(j) + '.jpg')

if __name__ == '__main__':
    main()
