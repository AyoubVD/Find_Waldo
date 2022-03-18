from imageRE import OpenPicture as P
from imageRE import Resizer as R
from cowsay import cheese, daemon, trex
from imageRE import Edits as E
#from imageRE import Resizer as R#

image = P.browseFiles()
cheese("Let's find Wally")
#P.changePath(image)
w,h = R.checkSize(image)
daemon("This takes so long")
R.addPadding(image, w, h)
trex("Padding has been removed")
E.mpTOwav("C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/mp3/thatshim.mp3")