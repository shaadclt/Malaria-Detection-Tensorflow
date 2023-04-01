from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import streamlit as st
import os

# Model saved with Keras model.save()
MODEL_PATH ='model_vgg19.h5'

# Loading the trained model
model = load_model(MODEL_PATH)

def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    print(img)

    # Preprocessing the image
    x = image.img_to_array(img)

    # Scaling
    x = x/255
    x = np.expand_dims(x, axis=0)
   
    x = preprocess_input(x)

    preds = model.predict(x)
    print(preds)
    preds = np.argmax(preds, axis=1)

    print(preds)
    
    if preds == 1:
        preds = "The Person is not Infected With Malaria"
    else:
        preds = "The Person is Infected With Malaria"
    
    return preds

def main():
    st.set_page_config(page_title="Malaria Detection App")

    st.title("Malaria Detection App")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        # Save the uploaded file to a temporary directory
        with open(os.path.join("tempDir", uploaded_file.name), "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_path = os.path.join("tempDir", uploaded_file.name)

        # Make a prediction on the uploaded file
        prediction = model_predict(file_path, model)

        # Display the uploaded image and the prediction result
        st.info(prediction)
        # st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        st.image(uploaded_file, caption="Uploaded Image", width=300)
        

if __name__ == "__main__":
    main()
