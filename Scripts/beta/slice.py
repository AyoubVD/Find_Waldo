from PIL import Image
from itertools import product
import os
import image_slicer
import shutil

#192

inp ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/"
outp ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/"
filename = "one.png"
d = 64

''' 
files = os.listdir(inp)
print(files)
for file in files:
    print(file)

 '''
p ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/one.png"

#image_slicer.main.slice(filename, number_tiles=64, col=None, row=None, save=True, DecompressionBombWarning=True)
def dicer(img):
    x = "C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/one.png"
    shutil.move(img, x)
    image_slicer.slice(x, 46)
''' def tile(filename, inp, outp, d):
    name, ext = os.path.splitext(filename)
    img = Image.open(os.path.join(inp, filename))
    w, h = img.size
    
    grid = product(range(0, h-h%d, d), range(0, w-w%d, d))
    for i, j in grid:
        box = (j, i, j+d, i+d)
        out = os.path.join(outp, f'{name}_{i}_{j}{ext}')
        img.crop(box).save(out)
        
tile(filename, inp, outp, d) 

def crop(path, inp, height, width, k, page, area):
    im = Image.open(inp)
    imgwidth, imgheight = im.size
    for i in range(0,imgheight,height):
        for j in range(0,imgwidth,width):
            box = (j, i, j+width, i+height)
            a = im.crop(box)
            try:
                o = a.crop(area)
                o.save(os.path.join(path,"PNG","%s" % page,"IMG-%s.png" % k))
            except:
                pass
            k +=1       
'''