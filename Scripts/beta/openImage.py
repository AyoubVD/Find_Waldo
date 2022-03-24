from imageRE import OpenPicture as P
from imageRE import Resizer as R
from cowsay import cheese, daemon, trex
#from imageRE import Edits as E
import os
import pydub
from pydub import AudioSegment
from pydub.playback import play
#from imageRE import Resizer as R#
from imageFormat import sliceIt
from findHim import findW
from findHim import fitW
import PIL

#43.75
#C:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo\Scripts\images\64\waldo
image = P.browseFiles()
print(image)
#image = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/full/madeb.png'
#cheese("Let's find Wally")
#P.changePath(image)
#w,h = R.checkSize(image)
print(PIL.Image.open(image).size)
daemon("I'm training the model, be patient!")
#findW()      
#os.rename("C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Models/modelW.pb","C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Models/modelW")  
for x in os.listdir('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Models/'):
        print(x)
print(fitW((image)))
#paddedimg = R.addPadding(image, w, h)  
#trex("Padding has been added")
''' sliceIt(paddedimg)
path = 'C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/padded/'
for narnia in os.listdir(path):
        paddedpath = path + narnia
        #findW(paddedpath) 
        width, height = PIL.Image.open(paddedpath).size
        #print(slices(paddedpath))
        #break
        print(width)
        print(height)    
        os.remove(paddedpath)    
#E.mpTOwav("C:/thatshim.mp3")
#print(os.getcwd())
#pydub.AudioSegment.ffmpeg = "C:/"   
#print(pydub.AudioSegment.ffmpeg)
#audiofile =  AudioSegment.from_mp3("thatshim.mp3")
#play(audiofile) '''