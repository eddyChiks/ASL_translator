"""
Definition of views.
"""

from datetime import datetime
from django.shortcuts import render
from django.http import HttpRequest
from django.template import loader
from django.http import StreamingHttpResponse
from django.http import HttpResponse
import os
import cv2
import threading
from django.views.decorators import gzip
from .models import *
import ctypes
import time
import numpy as np
import mediapipe as mp
import torch
from torchvision.transforms import ToTensor
from torch import nn
from .CNN import myCNN
from django.conf import settings

class VideoCamera(object):
    def __init__(self,count):
        self.video = cv2.VideoCapture(0)
        self.frame = self.video.read()[1]
        self.streaming = custom_thread("cam"+str(count),self)
        self.streaming.start()
        self.predicted = None

    def __del__(self):
        self.video.release()

    def get_frame(self):
        image = self.frame
        if image is not None:
            image,self.predicted = process_image(image)
            jpeg = cv2.imencode('.jpg', image)[1]
            return jpeg.tobytes()
        else:
            self.streaming.stop = True

    def update(self):
        try:
            ret,self.frame = self.video.read()
        except:
            pass
class custom_thread(threading.Thread):
    def __init__(self, name,cam):
        threading.Thread.__init__(self)
        self.name = name
        self.cam = cam
        self.stop = 0
        print(self.name+" created")
             
    def run(self):
 
        try:
            while True:
                self.cam.update()
                print(self.name+" is running")
                if self.stop:
                    print(self.name+ " end of run")
                    break
        except:
            pass
          
  

def home(request):
    """Renders the home page."""
    assert isinstance(request, HttpRequest)
    return render(
        request,
        'app/index.html',
        {
            'title':'Home Page',
            'year':datetime.now().year,
        }
    )
    
def signs(request):
    """Renders the about page."""
    assert isinstance(request, HttpRequest)
    dir = 'app/static/signs'
    labels = {0:'A', 1:'B', 2:'C',3:'D', 4:'del',5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N',15:'O', 16:'P', 17:'Q', 18:'R', 19:'S', 20:'space',21:'T', 22:'U', 23:'V', 24:'W', 25:'X', 26:'Y', 27:'Z'}
    images = os.listdir(dir)
    for i in range(len(images)):
        images[i]=(images[i],labels[i])
    return render(
        request,
        'app/signs.html',
        {
            'title':'Alphabet',
            'year':datetime.now().year,
            'images':images,
        }
    )

@gzip.gzip_page
def cam_stream(request):
    if 'count' not in request.session:
        request.session['count'] = 1
    else: 
        request.session['count']+=1
    count = request.session['count']
    test = cv2.VideoCapture(0)
    if not test or not test.isOpened():
        pass
    else:
        cam = VideoCamera(count)
        try: 
            if gen(cam) is not None:
                return StreamingHttpResponse(gen(cam), content_type = "multipart/x-mixed-replace;boundary=frame")
        except:
            pass

def gen(camera):
    while True:
        frame = camera.get_frame()
        if frame is not None:
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n\r\n')

def activate_cam(request):
    template = loader.get_template('app/index.html')
    context = {
        'activate':request.POST.get('activate')
    }
    if request.POST.get('deactivate'):
        for thread in threading.enumerate(): 
            if thread.name =="cam"+str(request.session['count']):
                cam = thread.cam
                thread.stop = True
                thread.join()
                cam.__del__()
                print(thread.name+" terminated!")

        for thread in threading.enumerate():
            print(thread.name+str(thread.is_alive()))
    else:
        test = cv2.VideoCapture(0)
        if not test or not test.isOpened():
            context['message'] = "Camera is not available!"
            context['activate'] = not context['activate']
    return HttpResponse(template.render(context,request))

def process_image(frame):
    labels = {0:'A', 1:'B', 2:'C',3:'D', 4:'del',5:'E', 6:'F', 7:'G', 8:'H', 9:'I', 10:'J', 11:'K', 12:'L', 13:'M', 14:'N',15:'O', 16:'P', 17:'Q', 18:'R', 19:'S', 20:'space',21:'T', 22:'U', 23:'V', 24:'W', 25:'X', 26:'Y', 27:'Z'}
    label=None
    mpHands = mp.solutions.hands
    mpDraw = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    with mpHands.Hands(
        static_image_mode=True,
        max_num_hands=1,
        min_detection_confidence=0.5) as mp_model:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.flip(frame,1)
        results = mp_model.process(frame)
        if results.multi_hand_landmarks is not None:
            for hand_landmarks in results.multi_hand_landmarks: 
                #getting bounding box coordinates
                brect = get_box(frame, hand_landmarks)
            if brect!=0:
                frame = cv2.rectangle(frame,(brect[0],brect[1]),(brect[2],brect[3]),(0, 0, 0), 1)
                new_img = np.ones((250,250,3),dtype=np.uint8)
                mpDraw.draw_landmarks(
                    new_img,
                    hand_landmarks,
                    mpHands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())
                rect = get_box(new_img,hand_landmarks)
                hand = new_img[rect[1]:rect[3],rect[0]:rect[2]]
                hand = np.array(hand)
                hand = cv2.resize(hand,(75,75))
                model = myCNN()
                model.load_state_dict(torch.load('app/static/ASL_model.pth'))
                model.eval()
                with torch.no_grad():

                    #converting hand image to tensor
                    transform = ToTensor()
                    test = transform(hand)
                    test = test.unsqueeze(0)
                
                    #prediction
                    pred = model(test)

                    #normalizing probabilities
                    soft = nn.Softmax(dim=1)
                    pred = soft(pred)
                    pred = np.argmax(pred)

                    #getting predicted letter
                    label = labels[pred.item()]
                    cv2.putText(frame,label,(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
        else:
            cv2.putText(frame,"No hand detected",(175,50),cv2.FONT_HERSHEY_COMPLEX,1,(255,0,0),3)
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    return frame,label

def get_box(image,landmarks):
    image_width, image_height = image.shape[1], image.shape[0]
    landmark_array = np.empty((0, 2), int)
    for _, landmark in enumerate(landmarks.landmark):
        landmark_x = min(int(landmark.x * image_width), image_width - 1)
        landmark_y = min(int(landmark.y * image_height), image_height - 1)
        landmark_point = [np.array((landmark_x, landmark_y))]
        landmark_array = np.append(landmark_array, landmark_point, axis=0)
    x, y, w, h = cv2.boundingRect(landmark_array)
    if x-10>0 and y-10>0 and x+w+10<image.shape[1] and y+h+10<image.shape[0]:
        return [x-10, y-10, x + w + 10, y + h +10]
    else:
        if x-10<=0:
            x-=x-10
        if y-10<=0:
            y-=y-10
        if x+w+10>=image.shape[1]:
            w-= x+w+10-image.shape[1]
        if y+h+10>=image.shape[0]:
            h-=y+h+10-image.shape[0]
        return [x-10, y-10, x + w + 10, y + h +10]

def get_translation(request):
    text = request.GET['current_text']
    predicted = None
    for thread in threading.enumerate(): 
        if thread.name =="cam"+str(request.session['count']):
            cam = thread.cam
            predicted = cam.predicted
    if predicted=="space":
        predicted=" "
    elif predicted=="del":
        text=""
        predicted=""
    text = text + predicted if predicted else text
    return HttpResponse(text)