import streamlit as st 
import cv2
import yaml 
import pickle 
from utils import *
import numpy as np
import os, datetime, json, sys, pathlib, shutil
import pandas as pd
from pandas.errors import EmptyDataError
from datetime import datetime

st.set_page_config(layout="wide")
st.title("Face Recognition App")
st.write("This app is used to add new faces to the dataset")
# Load admin credentials from a YAML file
admin_credentials = yaml.load(open('config.yaml', 'r'), Loader=yaml.FullLoader)
DATASET_DIR = admin_credentials['PATH']['DATASET_DIR']
PKL_PATH = admin_credentials['PATH']['PKL_PATH']
os.chmod(PKL_PATH, 0o666)

# Function to authenticate admin
def authenticate_admin(username, password):
    return username == admin_credentials['admin']['Username'] and password == admin_credentials['admin']['Password']

# Admin login page
def admin_login():
    st.title("Admin Login")
    username = st.text_input("Username",key="username_input")
    password = st.text_input("Password", type="password",key="password_input")
    if st.button("Login",key="login_button"):
        if authenticate_admin(username, password):
            st.success("Logged in successfully!")
            st.session_state.logged_in = True
            st.session_state.active_section = "main"
            
            st.empty()  # Clear content of the admin login page
            st.write('')
            return True
        else:
            st.error("Invalid username or password")
            return False


#Back to admin actions
def back_to_admin_actions():
    if st.button("Back to Admin Actions"):
        st.session_state.active_section = "main"

#Clear page
def clear_main_page():
    st.empty()


def view_attendace():
    st.markdown("visitor History")
    visitor_history_folder = 'visitor_history'
    f_p = os.path.join(visitor_history_folder, 'visitor_history.csv')
    
    if st.session_state.logged_in:   
        try:
            df_attendace = pd.read_csv(f_p)
        except pd.errors.EmptyDataError:
            # If the CSV file is empty, create an empty DataFrame
            df_attendace = pd.DataFrame(columns=['ID', 'Name', 'Time', 'Image'])
        
        # df_attendace = df_attendace.sort_values(by='Time',ascending=False)
        df_attendace.reset_index(inplace=True, drop=True)

        st.write(df_attendace)



# Main page
def main_page():
    st.header('Admin Dashboard')
    st.title("Admin Actions")
    if st.session_state.get('logged_in', False):
        if st.session_state.active_section == "main":
            if st.button("View Database", key="view_btn"):
                st.session_state.active_section = "view_database"
            if st.button("Visitor History", key="visitor_hstry"): 
                st.session_state.active_section = "view_attendance"
            if st.button("Update Database", key="update_btn"):
                st.session_state.active_section = "update_database"
            if st.button("Reset Database", key="reset_btn"):
                st.session_state.active_section = "reset_database"   
            if st.button("Logout", key="logout_btn"):
                logout()
        elif st.session_state.active_section == "view_database":
            view_database()
            back_to_admin_actions()
        elif st.session_state.active_section == "view_attendance":
            view_attendace()
            back_to_admin_actions()
        elif st.session_state.active_section == "update_database":
            update_database()
            back_to_admin_actions()
        elif st.session_state.active_section == "reset_database":
            reset_database()
            back_to_admin_actions()
    else:
        st.error("Please login to access this feature.")


#databse view
def view_database():
    
    st.title("Database View")
    if st.session_state.logged_in:
        # Load database 
        with open(PKL_PATH, 'rb') as file:
            database = pickle.load(file)

        Index, Id, Name, Image  = st.columns([0.5,0.5,3,3])

        for idx, person in database.items():
            with Index:
                st.write(idx)
            with Id: 
                st.write(person['id'])
            with Name:     
                st.write(person['name'])
            with Image:     
                st.image(person['image'], width=200)

#databse Update
def update_database():
    st.title("Update Database")
    if st.session_state.logged_in:
        menu = ["Adding","Deleting", "Adjusting"]
        choice = st.sidebar.selectbox("Options",menu)
        if choice == "Adding":
            name = st.text_input("Name",placeholder='Enter name')
            id = st.text_input("ID",placeholder='Enter id')
            #Create 2 options: Upload image or use webcam
            #If upload image is selected, show a file uploader
            #If use webcam is selected, show a button to start webcam
            upload = st.radio("Upload image or use webcam",("Upload","Webcam"))
            if upload == "Upload":
                uploaded_image = st.file_uploader("Upload",type=['jpg','png','jpeg'])
                if uploaded_image is not None:
                    st.image(uploaded_image)
                    submit_btn = st.button("Submit",key="submit_btn")
                    if submit_btn:
                        if name == "" or id == "":
                            st.error("Please enter name and ID")
                        else:
                            ret = submitNew(name, id, uploaded_image)
                            if ret == 1: 
                                st.success("User Added")
                            elif ret == 0: 
                                st.error("User ID already exists")
                            elif ret == -1: 
                                st.error("There is no face in the picture")
            elif upload == "Webcam":
                img_file_buffer = st.camera_input("Take a picture")
                submit_btn = st.button("Submit",key="submit_btn")
                if img_file_buffer is not None:
                    # To read image file buffer with OpenCV:
                    bytes_data = img_file_buffer.getvalue()
                    cv2_img = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
                    if submit_btn: 
                        if name == "" or id == "":
                            st.error("Please enter name and ID")
                        else:
                            ret = submitNew(name, id, cv2_img)
                            if ret == 1: 
                                st.success("User Added")
                            elif ret == 0: 
                                st.error("User ID already exists")
                            elif ret == -1: 
                                st.error("There is no face in the picture")
        elif choice == "Deleting":
            def del_btn_callback(id):
                deleteOne(id)
                st.success("User deleted")
                
            id = st.text_input("ID",placeholder='Enter id')
            submit_btn = st.button("Submit",key="submit_btn")
            if submit_btn:
                name, image,_ = get_info_from_id(id)
                if name == None and image == None:
                    st.error("User ID does not exist")
                else:
                    st.success(f"Name of user with ID {id} is: {name}")
                    st.warning("Please check the image below to make sure you are deleting the right user")
                    st.image(image)
                    del_btn = st.button("Delete",key="del_btn",on_click=del_btn_callback, args=(id,)) 
                
        elif choice == "Adjusting":
            def form_callback(old_name, old_id, old_image, old_idx):
                new_name = st.session_state['new_name']
                new_id = st.session_state['new_id']
                new_image = st.session_state['new_image']
                
                name = old_name
                id = old_id
                image = old_image
                
                if new_image is not None:
                    image = cv2.imdecode(np.frombuffer(new_image.read(), np.uint8), cv2.IMREAD_COLOR)
                    
                if new_name != old_name:
                    name = new_name
                    
                if new_id != old_id:
                    id = new_id
                
                ret = submitNew(name, id, image, old_idx=old_idx)
                if ret == 1: 
                    st.success("User Added")
                elif ret == 0: 
                    st.error("User ID already exists")
                elif ret == -1: 
                    st.error("There is no face in the picture")
            id = st.text_input("ID",placeholder='Enter id')
            submit_btn = st.button("Submit",key="submit_btn")
            if submit_btn:
                old_name, old_image, old_idx = get_info_from_id(id)
                if old_name == None and old_image == None:
                    st.error("User ID does not exist")
                else:
                    with st.form(key='my_form'):
                        st.title("Adjusting User info")
                        col1, col2 = st.columns(2)
                        new_name = col1.text_input("Name",key='new_name', value=old_name, placeholder='Enter new name')
                        new_id  = col1.text_input("ID",key='new_id',value=id,placeholder='Enter new id')
                        new_image = col1.file_uploader("Upload new image",key='new_image',type=['jpg','png','jpeg'])
                        col2.image(old_image,caption='Current image',width=400)
                        st.form_submit_button(label='Submit',on_click=form_callback, args=(old_name, id, old_image, old_idx))

#database reset 
def reset_database():
    st.title("Reset Database")
    if st.session_state.logged_in:
        with st.form(key='reset_form'):
            submit_button = st.form_submit_button(label='REBUILD DATASET')
            if submit_button:
                with st.spinner("Rebuilding dataset..."):
                    build_dataset()
                st.success("Dataset has been reset")


#logout
def logout():
    if st.session_state.logged_in:
        st.session_state.logged_in = False
        st.session_state.active_section = None
        st.success("Logged out successfully.")

def main():
    if 'logged_in' not in st.session_state or not st.session_state.logged_in:
        login_successful = admin_login()
        if login_successful:
            st.write("")
            st.empty()
            st.session_state.logged_in = True
            main_page()
    else:
        main_page()


if __name__ == "__main__":
    main()