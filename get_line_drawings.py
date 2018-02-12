import numpy as np
import cv2
from glob import glob
import os
import sys

data = glob(os.path.join("imgs", "*.jpg"))

def get_image(image_path, size=256):
    return transform(cv2.imread(image_path, 1), size)

def transform(image, size=256):
    cropped_image = cv2.resize(image, (size, size))
    return np.array(cropped_image)

if __name__ == "__main__":
    
    out_size = 256

    if len(sys.argv) > 1:
        if sys.argv[1] == "-o" :
            out_size = int(sys.argv[2])

    if not os.path.exists("out_imgs"):
        os.makedirs("out_imgs")    

    images = np.array([get_image(sample_file, out_size) for sample_file in data])
    line_drawings = np.array([cv2.adaptiveThreshold(cv2.cvtColor(ba, cv2.COLOR_BGR2GRAY), 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, blockSize=9, C=2) for ba in images])
    line_drawings = np.expand_dims(line_drawings, 3)
    output = np.tile(line_drawings,3)

    for i in range(len(data)):
        cv2.imwrite('out_' + data[i], output[i])

