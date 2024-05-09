import face_recognition as frg
import pickle as pkl 
import os, datetime, json, sys, pathlib, shutil
import cv2 
import numpy as np
import yaml
import streamlit as st 
import pandas as pd
from collections import defaultdict
from PIL import Image
from datetime import datetime
import csv

information = defaultdict(dict)
cfg = yaml.load(open('config.yaml','r'),Loader=yaml.FullLoader)
DATASET_DIR = cfg['PATH']['DATASET_DIR']
PKL_PATH = cfg['PATH']['PKL_PATH']

STREAMLIT_STATIC_PATH = pathlib.Path(st.__path__[0]) / 'static'
LOG_DIR = (STREAMLIT_STATIC_PATH / "logs")
if not LOG_DIR.is_dir():
    LOG_DIR.mkdir()
VISITOR_HISTORY = 'visitor_history'
if not os.path.exists(VISITOR_HISTORY):
    os.mkdir(VISITOR_HISTORY)
file_history = 'visitors_history.csv'
## Image formats allowed
allowed_image_type = ['.png', 'jpg', '.jpeg']

def get_databse():
    with open(PKL_PATH,'rb') as f:
        database = pkl.load(f)
    return database

def recognize(image, TOLERANCE):
    database = get_databse()
    known_encoding = [database[id]['encoding'] for id in database.keys()]
    name = 'Unknown'
    id = 'Unknown'
    face_locations = frg.face_locations(image)
    face_encodings = frg.face_encodings(image, face_locations)
    Time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        matches = frg.compare_faces(known_encoding, face_encoding, tolerance=TOLERANCE)
        distance = frg.face_distance(known_encoding, face_encoding)
        name = 'Unknown'
        id = 'Unknown'
        if True in matches:
            match_index = matches.index(True)
            name = database[match_index]['name']
            id = database[match_index]['id']
            distance = round(distance[match_index], 2)
            # Show pop-up message for access granted
            st.success(f'Access granted at {Time}', icon="✅")
            # Green color for verified
            cv2.rectangle(image, (left, top), (right, bottom),(0,255,0) , 2)
            cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0) , 2)
            # Save visitor log in visitor's image history folder
            save_visitor_log(image, name, id, Time)
        else:
            # Show pop-up message for access denied
            st.error(f'Access Denied at {Time}', icon="🚨")
            # Red color for denied
            cv2.rectangle(image, (left, top), (right, bottom),(0, 0, 255) , 2)
            cv2.putText(image, name, (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255) , 2)
            save_visitor_log(image, name, id, Time)
    return image, name, id, Time

def save_visitor_log(image, name, id, Time):
    visitor_history_folder = 'visitor_history'  # Path to the visitor history folder
    if not os.path.exists(visitor_history_folder):
        os.makedirs(visitor_history_folder)
    # Save image with visitor ID as filename
    visitor_image_filename = os.path.join('visitor_history', f'{id}.jpg')
    cv2.imwrite(visitor_image_filename, image)

    # Save visitor log in CSV file
    visitor_log_filename = os.path.join(visitor_history_folder, 'visitor_history.csv')
    write_header = not os.path.exists(visitor_log_filename)
    with open(visitor_log_filename, mode='a', newline='') as file:
        writer = csv.writer(file)
        if write_header:
            writer.writerow(['ID', 'Name', 'Time', 'Image'])  # Write header if it's a new file
        writer.writerow([id, name, Time, visitor_image_filename])

def isFaceExists(image): 
    face_location = frg.face_locations(image)
    if len(face_location) == 0:
        return False
    return True
def submitNew(name, id, image, old_idx=None):
    database = get_databse()
    #Read image 
    if type(image) != np.ndarray:
        image = cv2.imdecode(np.fromstring(image.read(), np.uint8), 1)

    isFaceInPic = isFaceExists(image)
    if not isFaceInPic:
        return -1
    #Encode image
    encoding = frg.face_encodings(image)[0]
    #Append to database
    #check if id already exists
    existing_id = [database[i]['id'] for i in database.keys()]
    #Update mode 
    if old_idx is not None: 
        new_idx = old_idx
    #Add mode
    else: 
        if id in existing_id:
            return 0
        new_idx = len(database)
    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
    database[new_idx] = {'image':image,
                        'id': id, 
                        'name':name,
                        'encoding':encoding}
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True
def get_info_from_id(id): 
    database = get_databse() 
    for idx, person in database.items(): 
        if person['id'] == id: 
            name = person['name']
            image = person['image']
            return name, image, idx       
    return None, None, None
def deleteOne(id):
    database = get_databse()
    id = str(id)
    for key, person in database.items():
        if person['id'] == id:
            del database[key]
            break
    with open(PKL_PATH,'wb') as f:
        pkl.dump(database,f)
    return True
def build_dataset():
    counter = 0
    for image in os.listdir(DATASET_DIR):
        image_path = os.path.join(DATASET_DIR,image)
        image_name = image.split('.')[0]
        parsed_name = image_name.split('_')
        person_id = parsed_name[0]
        person_name = ' '.join(parsed_name[1:])
        if not image_path.endswith('.jpg'):
            continue
        image = frg.load_image_file(image_path)
        information[counter]['image'] = image 
        information[counter]['id'] = person_id
        information[counter]['name'] = person_name
        information[counter]['encoding'] = frg.face_encodings(image)[0]
        counter += 1

    with open(os.path.join(DATASET_DIR,'database.pkl'),'wb') as f:
        pkl.dump(information,f)



if __name__ == "__main__": 
    deleteOne(4)

