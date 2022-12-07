from streamlit_webrtc import webrtc_streamer
import av
import streamlit as st
#import torch
#import os
#from PIL import Image
import cv2
import numpy as np
import ssl, json
import paho.mqtt.client as paho

#import torch

#from torchvision import models
import pytesseract # This is the TesseractOCR Python library
# Set Tesseract CMD path to the location of tesseract.exe file
#pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

#from paddleocr import PaddleOCR
#ocr = PaddleOCR(use_angle_cls=True)


# Model - we will use yolov5s
#model = torch.hub.load('./master/yolov5', 'custom', 'best.onnx')
#model = torch.hub.load('./master/yolov5', 'custom', path = 'best.pt', force_reload=True, source='local')
carplate_haar_cascade = cv2.CascadeClassifier('./alp_model.xml')
#model = cv2.dnn.readNetFromONNX('./best.onnx')
# ---------------------------------------MQTT Functions-----------------------------------------
# flag_connected = 0
# def on_connect(client, userdata, flags, rc):  
#     global flag_connected
#     flag_connected = 1 
#     print("Connected to the cloud!")

# def on_disconnect(client, userdata, rc):
#     global flag_connected
#     flag_connected = 0 
#     print("DISCONNECTED!")

#     # Defining the mqtt connection  
# client = paho.Client() 
# client.on_connect = on_connect
# client.on_disconnect = on_disconnect

#     # Setting the username password
# client.username_pw_set(username='ngalande', password='alprs@pappi')

#     # Connecting to the broker  
# client.tls_set(cert_reqs=ssl.CERT_NONE, tls_version=ssl.PROTOCOL_TLS)
# client.connect("4eb4a74af4a64aa7b440dd2d2451e924.s2.eu.hivemq.cloud", 8883)
#client.loop_forever()
#---------------------------------------------------------------------------------------------
# import VideoHTMLAttributes 
# muted = st.checkbox("Mute") 
# webrtc_streamer( key="mute_sample", video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}, muted=muted ), ) 




def carplate_detect(image):
        carplate_overlay = image.copy() 
        carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in carplate_rects: 
            cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,255,0), 2) 
            carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
            
            
        return carplate_overlay


# Create function to retrieve only the car plate region itself
def carplate_extract(image):
    carplate_img = image
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects: 
        carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
            
    return carplate_img

# Enlarge image for further processing later on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width, height)
    resized_image = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_image
# def ptModel(img):
#     results = model(img, size=640)
#     return results



def video_frame_callback(frame):
    st.title('Image upload demo')
    img = frame.to_ndarray(format="bgr24")
    plate_img = frame.to_ndarray(format="bgr24")
    #results = model(img, size=640)
    # Read car image and convert color to RGB
    #carplate_img = cv2.imread('./images/car_image.png')
    #carplate_img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    #plt.imshow(carplate_img_rgb)
    # Import Haar Cascade XML file for Russian car plate numbers
    detected_carplate_img = carplate_detect(img)
    
    # plate = ptModel(img)
    plate = carplate_extract(plate_img)
    # Convert image to grayscale
    carplate_extract_img_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    # Display extracted car license plate image
    # Apply median blur
    carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray,3) # kernel size 3
    # Display the text extracted from the car plate
    ocr_result = pytesseract.image_to_string(carplate_extract_img_gray_blur, 
                                  config = f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if(len(ocr_result)>5):
        print(ocr_result)
        #client.publish("alprs/plate", payload=ocr_result, qos=1)
    else:
        print('No Vehicle detected')                         
    # ocr_result = ocr.ocr(carplate_extract_img_gray_blur, cls=True)
    # print(ocr_result)
    return av.VideoFrame.from_ndarray(detected_carplate_img, format="bgr24")

muted = st.checkbox("Mute") 
webrtc_streamer(key="example", video_frame_callback=video_frame_callback)
# webrtc_streamer( key="mute_sample", video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}, muted=muted ), ) 


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
#st.markdown(hide_streamlit_style, unsafe_allow_html=True) 