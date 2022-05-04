import PIL
from itertools import product
import os
import image_slicer

#x = "imagePath"
def sliceIt(x):
    width, height = PIL.Image.open(x).size
    image_slicer.slice(x, width/slices(x))
    y = x.replace(x.split('/')[len(x.split('/'))-1], "")
    os.remove(x)
    
    ''' print(files)
    for file in files:
        print(file) '''
def slices(x):
    width, height = PIL.Image.open(x).size
    return SquareUp(width, 64)
    
     
def SquareUp(width,z):
    y = width / z
    if(y<0 or y ==0):
        PlusItOut(width,z)
    else:
        z= z*4
        print("Z: ",z)
        print("Y: ",y)
        SquareUp(width,z)

def PlusItOut(width,z):
    y = width / z
    if(y ==0):
        return z
    elif(y>0):
        ZeroItOut(width,z)
    else:
        z= z/2
        print("Z: ",z)
        print("Y: ",y)
        PlusItOut(width,z)
        
def ZeroItOut(width,z):
    y = width / z
    if(y<0 or y ==0):
        return z
    else:
        z+=1
        print("Z: ",z)
        print("Y: ",y)
        PlusItOut(width,z)
    

def greyZone(x):
    img = PIL.Image.open(x).convert('L')
    img.save(x)