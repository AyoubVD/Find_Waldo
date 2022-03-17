from PIL import Image
from itertools import product
import os
import image_slicer

#x = "imagePath"
def sliceIt(x):
    image_slicer.slice(x, 122)
    files = os.listdir(x)
    files.remove(x)
    print(files)
    for file in files:
        print(file)