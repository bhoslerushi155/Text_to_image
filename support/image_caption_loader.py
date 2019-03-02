import os
import numpy as np
from keras.preprocessing.image import img_to_array ,load_img

def load_image_and_text(image_width , image_height , images_path , texts_path):
    images=dict()
    texts=dict()
    result=[]

    for f in os.listdir(images_path):
        file_path = os.path.join(images_path , f)
        name = f.replace('.jpg' , '')
        images[name] = file_path

    for f in os.listdir(texts_path):
        file_path = os.path.join(texts_path , f)
        name = f.replace('.txt' , '')
        texts[name] = file_path

    for name , img_path in images.items():
        if name in texts:
            text = texts[name]
            image = image_to_array(load_img(img_path , target_size=(image_width , image_height)))
            image = (image.astype(np.float32)/255)*2 - 1
            result.append([image , text])

    return np.array(result)
