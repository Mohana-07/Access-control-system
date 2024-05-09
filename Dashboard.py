
import streamlit as st
import cv2
import face_recognition as frg
import yaml 
from utils import  *
# Path: code\app.py

st.set_page_config(layout="wide")
#Config
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
PICTURE_PROMPT = cfg['INFO']['PICTURE_PROMPT']
WEBCAM_PROMPT = cfg['INFO']['WEBCAM_PROMPT']



st.sidebar.header("Configuration Information")
st.sidebar.markdown("This system is used to control access for high security buildings developed By Harika")



#Create a menu bar
menu = ["Webcam"]
choice = st.sidebar.selectbox("Input type",menu)
#Put slide to adjust tolerance
TOLERANCE = st.sidebar.slider("Tolerance",0.0,1.0,0.5,0.01)
st.sidebar.info("Tolerance is the threshold for face recognition. The lower the tolerance, the more strict the face recognition. The higher the tolerance, the more loose the face recognition.")

#Infomation section 
st.sidebar.title("Verification Information")
name_container = st.sidebar.empty()
id_container = st.sidebar.empty()
name_container.info('Name: Unnknown')
id_container.success('ID: Unknown')

    
if choice == "Webcam":
    st.title("Face Recognition App")
    st.write(WEBCAM_PROMPT)
    #Camera Settings
    cam = cv2.VideoCapture(0)
    cam.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cam.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    FRAME_WINDOW = st.image([])
    
    while True:
        ret, frame = cam.read()
        if not ret:
            st.error("Failed to capture frame from camera")
            st.info("Please turn off the other app that is using the camera and restart app")
            st.stop()
        image, name, id, Time = recognize(frame,TOLERANCE)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #Display name and ID of the person
        
        name_container.info(f"Name: {name}")
        id_container.success(f"ID: {id}")
        # st.info(f"Time: {Time}")
        FRAME_WINDOW.image(image)

with st.sidebar.form(key='my_form'):
    st.title("Clear existing Dataset")
    submit_button = st.form_submit_button(label='REBUILD DATASET')
    if submit_button:
        with st.spinner("Rebuilding dataset..."):
            build_dataset()
        st.success("Dataset has been reset")