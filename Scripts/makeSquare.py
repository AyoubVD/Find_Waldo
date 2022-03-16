from PIL import Image

p ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/full/one.png"
notP ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/one_01_01.png"

width, height = Image.open(notP).size
if(height<width):
    x = width -height
    addPadHeight(width, x)
else:
    x = height -width
    addPadWith(x,height)

def addPadWidth(x, height):
    new_width = width + x
    new_height = height
    right = x/2
    left = x/2
    top = 0
    bottom = 0
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))
    
    result.save('squared.jpg')
    
def addPadHeight(width, x):
    new_width = width 
    new_height = height + x
    right = 0
    left = 0
    top = x/2
    bottom = x/2
    
    result = Image.new(image.mode, (new_width, new_height), (0, 0, 0))
    
    result.paste(image, (left, top))
    
    result.save('squared.jpg')