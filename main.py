'''
Here we record and save the data of the unknown face from the vide0 for the time frame for
which the unknown face was in the video
'''

import cv2
import face_recognition
import os
import streamlit as st

# Load the dataset of known faces
known_faces = []
known_names = {}

dataset_path = "./face"

for name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, name)
    if os.path.isdir(person_dir):
        for image_path in os.listdir(person_dir):
            image = face_recognition.load_image_file(os.path.join(person_dir, image_path))
            face_encoding = face_recognition.face_encodings(image)[0]
            known_faces.append(face_encoding)
            known_names[tuple(face_encoding)] = name

# Initialize the video capture
video_capture = cv2.VideoCapture(0)  # 0 for default webcam

# Create a function to recognize faces
def recognize_faces():
    ret, frame = video_capture.read()
    if not ret:
        return None
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, face_location in zip(face_encodings, face_locations):
        matches = face_recognition.compare_faces(known_faces, face_encoding)
        name = "Unknown"

        if True in matches:
            matched_indices = [i for i, match in enumerate(matches) if match]
            names = [known_names[tuple(known_faces[i])] for i in matched_indices]
            name = max(set(names), key=names.count)
            color = (0, 255, 0)  # Green for recognized faces
        else:
            color = (0, 0, 255)  # Red for unknown faces

        top, right, bottom, left = face_location
        cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

        if name == "Unknown":
            # Save the video segment with the unknown face
            start_time = video_capture.get(cv2.CAP_PROP_POS_MSEC)
            unknown_face_frames = []
            while True:
                ret, temp_frame = video_capture.read()
                if not ret:
                    break
                unknown_face_frames.append(temp_frame)
                if len(unknown_face_frames) > 30:  # Save 1 second of video
                    break
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter('unknown_face.mp4', fourcc, 30, (frame.shape[1], frame.shape[0]))
            for temp_frame in unknown_face_frames:
                out.write(temp_frame)
            out.release()

    return frame

# Streamlit app
st.title("Face Recognition App")

run = st.checkbox("Run Face Recognition")

if run:
    while True:
        frame = recognize_faces()
        if frame is None:
            break
        st.image(frame, channels="BGR")
else:
    st.write("Stop Face Recognition")