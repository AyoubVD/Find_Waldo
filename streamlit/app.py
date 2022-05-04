import streamlit as st
import os

st.title("Let's find Wally 🔎")
uploaded_file = None
formats = ['.png', '.jpeg', '.jpg', '.bmp', '.raw', '.tiff']

uploaded_file = st.file_uploader("Choose an image",help = "Insert a picture which you want to check wether or not Wally is in it.", type = formats)

if(uploaded_file!=None):
    st.write("This is the image you've chosen")
    st.image(uploaded_file)
hom_image = "wally.png"
st.image(hom_image)


print(os.getcwd())
    
#streamlit run c:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo\streamlit\app.py
# streamlit run c:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo\streamlit\app.py [ARGUMENTS] C:\Users\ayoub\OneDrive\TMM\Stage fase 3\Arinti\FindWaldo\FindWaldo
#streamlit run streamlit\app.py
# streamlit run "./streamlit/app.py"