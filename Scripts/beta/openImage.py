from imageRE import OpenPicture as P
from imageRE import Resizer as R
from cowsay import cheese as C
from cowsay import daemon as D
from cowsay import ghostbusters as G
from imageRE import Edits as E
#from imageRE import Resizer as R#

image = P.browseFiles()
C("Let's find Wally")
w,h = R.checkSize(image)
D("This takes so long")
R.addPadding(image, w, h)
G.("Padding has been removed")
E.officer("C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/mp3/that's him officer.mp3")