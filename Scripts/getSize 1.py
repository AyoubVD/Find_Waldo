from PIL import Image

p ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/full/one.png"
notP ="C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/sliced/one/one_01_01.png"

width, height = Image.open("C:/wally.png").size
print(width)
#16
print(height)
#12
print(16*12)
def check_width(width):
    if(width%64==0):
        return True
    else:
        return False
    
def check_height(height):
    if(height%64==0):
        return True
    else:
        return False

print(check_width(width))
print(check_height(height))


