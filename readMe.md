# Project: Face Recognition App

This project implements a face recognition system with two functionalities:

## 1} Image Recognition

Upload an image to identify the person based on a dataset of known faces.
If recognized, the name is displayed.
If not recognized, add the face to the known faces database with a name (optional).

## 2} Live Video Recognition

Continuously identify faces in a live video stream using your webcam.
Recognized faces from the known faces dataset display their names.
When an unknown face is detected, a short video clip (around 1 second) is saved for further analysis (potentially for adding the face to the known faces database later).

## Requirements:

. Python 3+
. Libraries (specified in requirements.txt):
. face_recognition
. streamlit (https://docs.streamlit.io/) (for user interface)
. opencv-python (for live video processing, if applicable)

## Installation:

1} Clone this repository:

```git clone https://<your_github_username>/Project-v.git```

2} Navigate to the project directory:

```cd final_cv```

3} Install the required libraries:

```pip install -r requirements.txt```

## Usage:

### Image Recognition:

. Run the script:

```streamlit run a.py```

. Upload an image using the Streamlit interface.

. The app will display the recognition result (match or unknown).

### Live Video Recognition:

. Run the script:

```stremlit run main.py```

. The Streamlit interface will display the live video stream with recognized faces and their names.

## Data Structure:

. ```known_faces```: This folder stores images of known individuals used for recognition.
. ```unknown_face.mp4``` (optional): This file (created by main.py) stores a short video clip of an unknown face for potential later analysis.

## Additional Notes:

This project uses the face_recognition library, which has limitations. For more advanced face recognition tasks, explore deep learning libraries like TensorFlow or PyTorch.
Be mindful of potential privacy concerns when using face recognition technology.