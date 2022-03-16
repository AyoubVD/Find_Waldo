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
        
        # Change label contents
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
'''    
class Resizer(x):
    def checkSize(x):
        paddingW
        paddingH
        width, height = Image.open(x).size
        tempW = width
        tempH = height
        
        if(width>height):
            if(width%64==0):
                paddingW = width%64
            elif(width%64!=0):
                while(tempW!=0):
                    tempW+=1
                    tempW = tempW%64
            paddingW = tempW - width
            paddingH = tempH -width
        else:
            if(height%64==0):
                tempH = height%64
            elif(height%64!=0):
                while(tempH!=0):
                    tempH+=1
                    tempH = height%64
            paddingW = tempW - height
            paddingH = tempH - height
        return paddingW,paddingH
    #def addPadding(w,h): '''
        

                