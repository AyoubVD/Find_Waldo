from PIL import Image
import cv2
from slice import dicer
p ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/full/one.png"
notP ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/one_01_01.png"



def addPadding(img):
    image = Image.open(img)
    width, height = image.size
    x = int((3360-width)/2)
    y = int((3360-height)/2)
    right = x
    left = x
    top = y
    bottom = y
    
    
    new_width = 3360
    new_height = 3360
    
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))
    s = "C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/padded/one.png"
    result.save(s)
    dicer(s)
''' 
def addPadWidth(x, height):
    new_width = width + x
    new_height = height
    right = x/2
    left = x/2
    top = 0
    bottom = 0
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))
    
    result.save('squared.png')
    
def addPadHeight(width, x):
    new_width = width 
    new_height = height + x
    right = 0
    left = 0
    top = x/2
    bottom = x/2
    
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))
    
    result.save('squared.jpg') '''