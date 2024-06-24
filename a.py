'''
Here we can match the face of a person by uploading its image if the image is in known face list it shows Match found [name]
and if the match is not found it show no match found and a chec kbox to add this face to database and then label the face.
'''


import streamlit as st
import face_recognition
from PIL import Image, UnidentifiedImageError
import numpy as np
import os

# Load known faces
@st.cache_resource
def load_known_faces(known_faces_dir):
    known_face_encodings = []
    known_face_names = []

    for file_name in os.listdir(known_faces_dir):
        image_path = os.path.join(known_faces_dir, file_name)
        try:
            # Check if the file is a valid image
            with Image.open(image_path) as img:
                img.verify()  # Verify that it is an image

            image = face_recognition.load_image_file(image_path)
            face_encodings = face_recognition.face_encodings(image)

            if face_encodings:
                known_face_encodings.append(face_encodings[0])
                known_face_names.append(os.path.splitext(file_name)[0])
        except (UnidentifiedImageError, OSError) as e:
            st.warning(f"Cannot identify image file: {file_name}. Skipping. Error: {e}")
        except Exception as e:
            st.error(f"Error processing file {file_name}: {e}")

    return known_face_encodings, known_face_names


known_faces_dir = "./face"
unknown_faces_dir = "./unknown"
if not os.path.exists(unknown_faces_dir):
    os.makedirs(unknown_faces_dir)

known_face_encodings, known_face_names = load_known_faces(known_faces_dir)

# Streamlit app
st.title("Face Recognition App")
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        # Load uploaded image
        uploaded_image = Image.open(uploaded_file)
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Convert uploaded image to face_recognition format
        uploaded_image_np = np.array(uploaded_image)
        uploaded_face_encodings = face_recognition.face_encodings(uploaded_image_np)

        if uploaded_face_encodings:
            match_found = False
            for uploaded_face_encoding in uploaded_face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, uploaded_face_encoding)
                face_distances = face_recognition.face_distance(known_face_encodings, uploaded_face_encoding)
                best_match_index = np.argmin(face_distances)

                if matches[best_match_index]:
                    match_found = True
                    name = known_face_names[best_match_index]
                    st.success(f"Match found: {name}")
                    break

            if not match_found:
                st.warning("No match found.")
                add_unknown = st.checkbox("Add this face to the known faces database")

                if add_unknown:
                    new_name = st.text_input("Enter the name of the person:")

                    if new_name:
                        known_face_encodings.append(uploaded_face_encodings[0])
                        known_face_names.append(new_name)

                        # Saving the image in the known_faces directory
                        known_image_path = os.path.join(known_faces_dir, f"{new_name}.jpg")
                        uploaded_image.save(known_image_path)
                        st.success(f"Added {new_name} to the known faces database.")

                        # Updating the known faces list
                        known_face_encodings, known_face_names = load_known_faces(known_faces_dir)
                    else:
                        st.error("Please enter a name.")
        else:
            st.error("No face detected in the uploaded image.")
    except Exception as e:
        st.error(f"Error processing the uploaded image: {e}")
