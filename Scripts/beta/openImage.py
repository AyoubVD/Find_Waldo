from imageRE import OpenPicture as P
from imageRE import Resizer as R
from cowsay import cheese, daemon, trex
from imageRE import Edits as E
import pydub
from pydub import AudioSegment
import os
from pydub.playback import play
#from imageRE import Resizer as R#

image = P.browseFiles()
cheese("Let's find Wally")
#P.changePath(image)
w,h = R.checkSize(image)
daemon("This takes so long")
R.addPadding(image, w, h)
trex("Padding has been removed")
#E.mpTOwav("C:/thatshim.mp3")
print(os.getcwd())
pydub.AudioSegment.ffmpeg = "C:/"   
print(pydub.AudioSegment.ffmpeg)
audiofile =  AudioSegment.from_mp3("thatshim.mp3")
play(audiofile)