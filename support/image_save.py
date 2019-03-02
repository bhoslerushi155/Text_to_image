import numpy as np
import math
from PIL import Image

def image_for_snapshot(generated_images):
    count = generated_images.shape[0]
    width = int(math.sqrt(count))
    height = int(math.ceil(float(count)/float(width)))
    shape = generated_images.shape[1:]
    images = np.zeros((height * shape[0] , width * shape[1] , shape[2]) , dtype=generated_images.dtype)

    ind=0
    for img in generated_images:
        i = int(ind / width)
        j = ind % width
        images[i * shape[0]:(i + 1) * shape[0] , j * shape[1]:(j + 1)*shape[1] , :]=img
        ind += 1


def image_from_array(normalized_image_array):
    image = image_array*127.5 + 127.5
    img = Image.fromarray(image.astype(np.unit8))
    return img
