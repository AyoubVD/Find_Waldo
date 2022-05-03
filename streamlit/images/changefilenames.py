import os
print(os.getcwd())
PATHW = "./Stage/Arinti/FindWaldo/FindWaldo/streamlit/images/wally/"
PATHN = "./Stage/Arinti/FindWaldo/FindWaldo/streamlit/images/nwally/"
filesw = os.listdir(PATHW)
filesnw = os.listdir(PATHN)

count1 = len(filesw)
count2 = len(filesnw)

print(f"Before Renaming: {filesw}")
for i in range(len(filesw)):
   os.rename(PATHW+filesw[i], f"{PATHW}w{i+1}.jpg")
print(f"After Renaming: {os.listdir(PATHW)}")

print(f"Before Renaming: {filesnw}")
for i in range(len(filesnw)):
   os.rename(PATHN+filesnw[i], f"{PATHN}n{i+1}.jpg")
print(f"After Renaming: {os.listdir(PATHN)}")