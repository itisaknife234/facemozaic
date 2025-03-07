import numpy as np
import cv2
import io
import streamlit as st

class FaceMosaic:
    def __init__(self, face_cascade_path=None, eye_cascade_path=None):
        self.face_cascade = cv2.CascadeClassifier(
            face_cascade_path or cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        self.eye_cascade = cv2.CascadeClassifier(
            eye_cascade_path or cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
    
    def load_image(self, file_bytes):
        """ Load image from uploaded file bytes. """
        ff = np.frombuffer(file_bytes, dtype=np.uint8)
        return cv2.imdecode(ff, cv2.IMREAD_UNCHANGED)
    
    def detect_faces(self, image):
        """ Detect faces in the given image. """
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
        return faces
    
    def apply_mosaic(self, image, faces, mosaic_ratio=20):
        """ Apply mosaic effect to detected faces. """
        for (x, y, w, h) in faces:
            face_img = image[y:y+h, x:x+w]
            small_face = cv2.resize(face_img, (w//mosaic_ratio, h//mosaic_ratio), interpolation=cv2.INTER_LINEAR)
            mosaic_face = cv2.resize(small_face, (w, h), interpolation=cv2.INTER_NEAREST)
            image[y:y+h, x:x+w] = mosaic_face
        return image
    
    def process_uploaded_image(self, file_bytes):
        """ Load uploaded image, detect faces, apply mosaic, and return processed image. """
        image = self.load_image(file_bytes)
        faces = self.detect_faces(image)
        processed_image = self.apply_mosaic(image, faces)
        
        _, encoded_image = cv2.imencode('.jpg', processed_image)
        return io.BytesIO(encoded_image.tobytes())

st.title("Face Mosaic App")
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    face_mosaic = FaceMosaic()
    processed_image = face_mosaic.process_uploaded_image(uploaded_file.read())
    
    st.image(processed_image, caption="Processed Image", use_column_width=True)
    st.download_button(label="Download Processed Image", data=processed_image, file_name="mosaic.jpg", mime="image/jpeg")
