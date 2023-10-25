from streamlit_webrtc import webrtc_streamer, RTCConfiguration
import av
import streamlit as st
import cv2
import re

from paddleocr import PaddleOCR
PaddleOCR(show_log = False) 
ocr = PaddleOCR(use_angle_cls=True, lang='en')

carplate_haar_cascade = cv2.CascadeClassifier('./alp_model.xml')
hide_streamlit_style = """
            <style>
            [data-testid="stToolbar"] {visibility: hidden !important;}
            footer {visibility: hidden !important;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 
st.title("Licence Plate Recognition")

def carplate_detect(image, ocr_plate):
        carplate_overlay = image.copy() 
        carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1, minNeighbors=5)
        for x,y,w,h in carplate_rects: 
            cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,255,0), 2) 
            carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
                    # Draw the detected license plate text
            cv2.putText(carplate_overlay, 'Licence Plate: '+ str(ocr_plate), (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

            
            
        return carplate_overlay


# Create function to retrieve only the car plate region itself
def carplate_extract(image):
    carplate_img = image
    carplate_overlay = image.copy() 
    carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1, minNeighbors=5)
    for x,y,w,h in carplate_rects: 
        cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,255,0), 1) 
        carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extract specific region of interest i.e. car license plate
        # cv2.imwrite('carplate_with_bbox.png', carplate_overlay)
            
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
    
    # plate = ptModel(img)
    plate = carplate_extract(plate_img)
    # Convert image to grayscale
    carplate_extract_img_gray = cv2.cvtColor(plate, cv2.COLOR_RGB2GRAY)
    # Display extracted car license plate image
    # Apply median blur
    # carplate_extract_img_gray_blur = cv2.medianBlur(carplate_extract_img_gray,1) # kernel size 3
    # Display the text extracted from the car plate
    # ocr_result = pytesseract.image_to_string(carplate_extract_img_gray_blur, lang='eng',
    #                               config = '--psm 10 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789')
    # pattern = r'[A-Z0-9]+'

    # # Apply the regular expression to the OCR result
    # filtered_result = re.findall(pattern, ocr_result)

    # # Join the matched characters and digits to form the cleaned result
    # cleaned_result = ''.join(filtered_result)
    result = ocr.ocr(carplate_extract_img_gray)
    ocr_result = ''
    for idx in range(len(result)):
        res = result[idx]
        for line in res:
            # print(line[0])
            ocr_result = line[0]
    # print(cleaned_result)
    if(len(ocr_result)>5):
        
        detected_carplate_img = carplate_detect(img, ocr_result)
        print('[DETECTED PLATE]: ', ocr_result)

    else:
        detected_carplate_img = carplate_detect(img, 'Unable to perform OCR')
        print('No Vehicle detected')                         
    # ocr_result = ocr.ocr(carplate_extract_img_gray_blur, cls=True)
    # print(ocr_result)
    return av.VideoFrame.from_ndarray(detected_carplate_img, format="bgr24")


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


