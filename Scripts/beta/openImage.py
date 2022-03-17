from imageRE import OpenPicture as P
from imageRE import Resizer as R
from cowsay import cheese as C
from cowsay import daemon as D
#from imageRE import Resizer as R#

image = P.browseFiles()
C("Let's find Wally")
w,h = R.checkSize(image)
D("This takes so long")
R.addPadding(image, w, h)