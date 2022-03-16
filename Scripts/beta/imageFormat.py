from PIL import Image
from itertools import product
import os
import image_slicer

inp = "imagePath"

image_slicer.slice(inp, 122)
files = os.listdir(inp)
files.remove(inp)
print(files)
for file in files:
    print(file)