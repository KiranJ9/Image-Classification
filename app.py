import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imread #To read the image
from skimage.transform import resize #Resizing the image
import pickle 
from PIL import Image
from pyngrok import ngrok
import streamlit as st
#st.set_option('deprecation.showFileUploaderEncoding', False)
st.title('Predict the image')
st.text('Upload your image')

model = pickle.load(open('img_model.p','rb'))

uploaded_file = st.file_uploader("Choose an image",type = "jpg")
if uploaded_file is not None:
  img = Image.open(uploaded_file)
  st.image(img,caption = "Uploaded image")
  if st.button("Predict"):
    st.write('Prediction...')
    CATEGORIES = ['burger','pizza','tasty pasta']
    flat_data = []
    img = np.array(img)
    resized_img = resize(img,(100,100,3))
    flat_data.append(resized_img.flatten())
    flat_data = np.array(flat_data)
    print(img.shape)
    y_out = model.predict(flat_data)
    y_out = CATEGORIES[y_out[0]]
    st.title(f'The image is a :{y_out}')
    q = model.predict_proba(flat_data)
    for index, item in enumerate(CATEGORIES):
      st.write(f'{item} : {q[0][index]*100}%')