from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import streamlit as st
import requests
#import torch
#import os
#from PIL import Image
import cv2
import numpy as np
import ssl, json
import paho.mqtt.client as paho
# import pika
#import torch
import os

#from torchvision import models
import pytesseract # This is the TesseractOCR Python library

carplate_haar_cascade = cv2.CascadeClassifier('./alp_model.xml')
#model = cv2.dnn.readNetFromONNX('./best.onnx')
# ---------------------------------------MQTT Functions-----------------------------------------
flag_connected = 0

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


def video_frame_callback(frame):
    st.title('Image upload demo')
    img = frame.to_ndarray(format="bgr24")
    plate_img = frame.to_ndarray(format="bgr24")
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
                                  config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    if(len(ocr_result)>6):
        
        id = '35e10339-b297-4f25-a915-f1c7179d99a1'
        payload = {
            "amount":"20",
            "number_plate": ocr_result
            }
        requests.post("https://etollapi.samwaku.com/api/v1.1/transactions/numberplate-transaction/35e10339-b297-4f25-a915-f1c7179d99a1", payload)
        print('[DETECTED PLATE]: ', ocr_result)
        # channel.basic_publish(exchange='',
        #               routing_key='etolldata',
        #               body=ocr_result)

        # print(" [x] Data Sent")
        # connection.close()
        # client.publish("alprs/plate", payload=ocr_result, qos=1)
    else:
        print('No Vehicle detected')                         
    # ocr_result = ocr.ocr(carplate_extract_img_gray_blur, cls=True)
    # print(ocr_result)
    return av.VideoFrame.from_ndarray(detected_carplate_img, format="bgr24")

muted = st.checkbox("Mute") 
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)
webrtc_streamer(
    key="pappi", 
    video_frame_callback=video_frame_callback,
    rtc_configuration=RTC_CONFIGURATION,
        media_stream_constraints={
            "video": True,
            "audio": False
        }
    )
# webrtc_streamer( key="mute_sample", video_html_attrs=VideoHTMLAttributes( autoPlay=True, controls=True, style={"width": "100%"}, muted=muted ), ) 


hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 