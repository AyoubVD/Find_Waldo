import PIL
from PIL import ImageTk
from tkinter import *
from tkinter import filedialog

class OpenPicture():
    def browseFiles():
        filename = filedialog.askopenfilename(initialdir = "/",
                                            title = "Find Wally",
                                            filetypes = (("Images",
                                                            "*.png*"),
                                                        ("Images",
                                                            "*.jpg*"),
                                                        ("All files",
                                                            "*.*"),
                                                        ("Images",
                                                            "*.jfif*")))
        return filename
    notP ="C:/wally.png"
    width, height = PIL.Image.open(notP).size
                                                                                                    
    # Create the root window
    window = Tk()
    pushingP = str(width)+"x"+str(height)
    window.title('Find Wally')
    window.geometry(pushingP)

    bg= ImageTk.PhotoImage(file="C:/wally.png")
    canvas= Canvas(window,width= width, height= height)
    canvas.pack(expand=True, fill= BOTH)
    canvas.create_image(0,0,image=bg, anchor="nw")

    btn = Button(window, text='Find a Wally picture', width=20,
                height=3, bd='10', command=browseFiles)
    btn.place(x=250, y=400)

class Resizer():
    def checkSize(x):
        paddingW = 0
        paddingH = 0
        width, height = PIL.Image.open(x).size
        tempW = width
        tempH = height
            
        if(width>height):
            if(width%64==0):
                paddingW = width%64
            elif(width%64!=0):
                while(tempW!=0):
                    tempW+=1
                    tempW = tempW%64
            paddingH = width - tempH
        else:
            if(height%64==0):
                tempH = height%64
            elif(height%64!=0):
                while(tempH!=0):
                    tempH+=1
                    tempH = height%64
            paddingW = height - tempW
        return paddingW,paddingH
    def addPadding(impath,w,h):
        #impath = "C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/full/one.png"
        w,h = Resizer.checkSize(impath)
        image = PIL.Image.open(impath)
        filepathname = impath.split('/')
        filename = filepathname[len(filepathname)-1].split('.')[0]
        print(filename)
        if(w==0):
            top = int(h/2)
            bottom = int(h/2)
            right = 0
            left = 0
        else:
            top = 0
            bottom = 0
            right = int(w/2)
            left = int(w/2)
        
        width, height = image.size
        
        new_width = int(width + right + left)
        new_height = int(height + top + bottom)
        
        result = PIL.Image.new(image.mode, (new_width, new_height), (0, 0, 0))
        
        result.paste(image, (left, top))
        
        result.save('C:/Users/ayoub/OneDrive/TMM/Stage fase 3/Arinti/FindWaldo/FindWaldo/Scripts/images/padded/'+filename+'_padded.png')