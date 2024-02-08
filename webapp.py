import streamlit as st
from PIL import Image
import numpy as np
import tensorflow.keras as keras
import matplotlib.pyplot as plt
import tensorflow_hub as hub
import tensorflow_hub as hub
import cv2
import numpy as np

st.title("Potato Leaf Disease Prediction")

def main() :
    file_uploaded = st.file_uploader("Choose an image...", type = "jpg")
    if file_uploaded is not None :
        image = Image.open(file_uploaded)
        st.write("Uploaded Image.")
        figure = plt.figure()
        plt.imshow(image)
        plt.axis("off")
        st.pyplot(figure)
        result, confidence = predict_class(image)
        st.subheader("Prediction:")
        st.write(result)
        st.subheader("Confidence:")
        st.write("{:.2f}%".format(confidence))
        st.subheader("Recommendations:")
        if result == "Potato__healthy":
            st.write("Good job! The potato leaf is perfectly healthy.")
        elif result=="Potato__Early_blight":
            st.write("• Maintain a proper crop rotation schedule.")
            st.write("• Keep the area around your plants clean and weed-free.")
            st.write("• Inspect your plants regularly for signs of disease or pests.")
            st.write("• Use disease-resistant potato varieties if possible.")
        elif result=="Potato__Late_blight":
            st.write("• Remove and destroy any infected plants immediately")
            st.write("• Avoid overhead irrigation")
            st.write("• Late blight can survive in the soil, so it's important to rotate crops to prevent the disease from building up in the soil.")
            st.write("• Late blight can spread quickly, so it's important to monitor the plants closely for any signs of the disease and take action immediately if it's detected.")
        else:
            st.write("Some error occured")

def predict_class(image) :
    with st.spinner("Loading Model..."):
        classifier_model = keras.models.load_model("super2.h5", custom_objects={"KerasLayer": hub.KerasLayer})

    shape = (224, 224, 3)  # Change the input shape to (224, 224, 3)
    model = keras.Sequential([hub.KerasLayer(classifier_model, input_shape=shape)])   
    test_image = image.resize((224, 224))
    test_image = keras.preprocessing.image.img_to_array(test_image)
    test_image /= 255.0
    test_image = np.expand_dims(test_image, axis=0)
    class_name = ["Potato__Early_blight", "Potato__Late_blight","Potato__healthy"]

    prediction = model.predict(test_image)
    confidence = round(100 * np.clip(np.max(prediction[0]), 0, 1), 2) #np.clip ranges the confidence from 0 to 1
    final_pred = class_name[np.argmax(prediction)]
    return final_pred, confidence

footer = """
<style>
a:link , a:visited {
    color: white;
    background-color: transparent;
    text-decoration: None;
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    padding: 1rem;
    background-color: #f5f5f5;
    color: #000;
    text-align: center;
    border-top: 1px solid #e5e5e5;
}
</style>

<div class="footer">
    Created by Dhaval Mainkar
</div>
"""

st.markdown(footer, unsafe_allow_html=True)


if __name__ == "__main__" :
    main()
