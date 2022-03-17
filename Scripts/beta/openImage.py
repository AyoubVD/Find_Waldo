from imageRE import OpenPicture as P
from imageRE import Resizer as R
#from imageRE import Resizer as R#
from PIL import Image

image = P.browseFiles()
im = Image.open(image)
  
im.show()