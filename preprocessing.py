import cv2 as cv
import numpy as np
from torchvision import transforms


def preprocess(image) :
    transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    square_image = pad_to_square(image)
    resized_image = cv.resize(square_image, (28, 28), interpolation = cv.INTER_AREA)
    cv.imwrite('preprocessed_image.png', resized_image)
    final_image = transform(resized_image)
    final_image = final_image.unsqueeze(0)

    return final_image
    
def pad_to_square(image) :
    height, width = image.shape[:2]
    position_x = int((max(width, height) - width) / 2)
    position_y = int((max(width, height) - height) / 2)
    new_image = np.zeros(shape = (max(width, height), max(width, height)), dtype = np.uint8)
    new_image[position_y : height + position_y, position_x : width + position_x] = image
    return new_image
    