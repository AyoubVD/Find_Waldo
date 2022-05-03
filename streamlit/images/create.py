#{"image_url": "AmlDatastore://workspaceblobstore/images/milk_bottle/66.jpg", "label": "milk_bottle"}
import os
PATHW = "F:/Arinti/Stage/Arinti/FindWaldo/FindWaldo/streamlit/images/wally/"
PATHN = "F:/Arinti/Stage/Arinti/FindWaldo/FindWaldo/streamlit/images/nwally/"
filesw = os.listdir(PATHW)
filesnw = os.listdir(PATHN)
names = filesw + filesnw
print(names)
PATH = "F:/Arinti/Stage/Arinti/FindWaldo/FindWaldo/streamlit/images/"
f = open(PATH+'trainwanno.jasonl', 'w')
with open('trainwanno.jasonl', 'w') as f:
    for x in names:
        if(x[0]=='w'):
            f.write('{"image_url": "AmlDatastore://workspaceblobstore/images/wally/' + x + '", "label": "wally"}\n')
        else:
            f.write('{"image_url": "AmlDatastore://workspaceblobstore/images/nwally/' + x + '", "label": "not wally"}\n')

