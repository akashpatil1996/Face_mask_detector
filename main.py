import numpy as np
from tensorflow.keras.models import load_model
import streamlit as st
from PIL import Image
import time

model = load_model('FaceMaskDetModel.h5')

st.title('Welcome to Face Mask Detector 😷')

uploaded_file = st.file_uploader('Upload an image, and the ML model will detect if the person is wearing mask or not')

if uploaded_file is not None:
  image = Image.open(uploaded_file)
  progress_text = "Predicting. Please wait."
  my_bar = st.progress(0, text=progress_text)

  for percent_complete in range(100):
    time.sleep(0.01)
    my_bar.progress(percent_complete + 1, text=progress_text)
  my_bar.empty()
  st.write("Done! Scroll down to see the result!")

  st.image(image, caption='Uploaded Image', use_column_width=True)
  image = image.resize((128, 128))
  image = image.convert('RGB')
  image = np.array(image)
  image = image / 255.0
  image = np.expand_dims(image, axis=0)
  pred = model.predict(image)
  pred_label = np.argmax(pred)
  if pred_label == 0:
    st.warning("The person is not wearing a mask")
  else:
    st.success("The person is wearing a mask")