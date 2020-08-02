from django.shortcuts import render , redirect
from django.http import HttpResponse
from django.http import request
from django.contrib.auth.forms import UserCreationForm
from django.contrib import messages
import cv2
from django.core.files.storage import FileSystemStorage
# Create your views here.

import numpy as np
import tensorflow as tf
from tensorflow import keras
import keras
from tensorflow.keras.applications import vgg16, inception_v3, resnet50, mobilenet
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.applications.imagenet_utils import decode_predictions
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
import matplotlib.pyplot as plt
#import cv2 as cv
import os
import tensorflow as tf
from pages.models import labels
from datetime import datetime
import cvlib as cv
def gender(path):
    image=cv2.imread(path)
    if image is None:
        print("Could not read input image")
        exit()
    model = load_model("models/gender_detection.model")
    
    face, confidence = cv.detect_face(image)

    classes = ['man','woman'] 
    for idx, f in enumerate(face):      
        (startX, startY) = f[0], f[1]
        (endX, endY) = f[2], f[3]
        cv2.rectangle(image, (startX,startY), (endX,endY), (0,255,0), 2)
        face_crop = np.copy(image[startY:endY,startX:endX])
        face_crop = cv2.resize(face_crop, (96,96))
        face_crop = face_crop.astype("float") / 255.0
        face_crop = img_to_array(face_crop)
        face_crop = np.expand_dims(face_crop, axis=0)
        conf = model.predict(face_crop)[0]
        #print(conf)
        #print(classes)
        idx = np.argmax(conf)
        label = classes[idx]
        return label

def color_pred(test_image):
    color_model = load_model('models/color.h5')
    result_color = color_model.predict_classes(test_image)
    color_classes = ['Black', 'Blue', 'Brown', 'Cyan', 'Gray', 'Green', 'More than 1 color', 'Orange', 'Pink', 'Purple', 'Red', 'White', 'Yellow']
    return(color_classes[int(result_color)])
def pattern_pred(test_image):
    pattern_model = load_model('models/pattern.h5')
    result_pattern = pattern_model.predict_classes(test_image)
    pattern_classes = ['Floral','Graphics','Plaid','Solid','Spotted','Striped']            
    prediction_pattern = pattern_classes[int(result_pattern)]
    #print(prediction_pattern)
    return(prediction_pattern)
def features(path):
    valid_classes = {
     'T-shirt': ['jersey', 'T-shirt', 'tee shirt'], 
     'Dress':['dress', 'gown', 'overskirt', 'hoopskirt', 'stole', 'abaya', 'academic_gown', 'poncho', 'breastplate'],
     'Outerwear':['jacket', 'raincoat', 'trench coat','book jacket', 'dust cover', 'dust jacket', 'dust wrapper', 'pitcher'], 
     'Suit':['suit','bow tie', 'bow-tie', 'bowtie','suit of clothes'], 'Shirt':['shirt'], 
     'Sweater':['sweater', 'sweatshirt','bulletproof_vest', 'velvet'] , 
     'Tank top':['blause', 'tank top', 'maillot', 'bikini', 'two-piece', 'swimming trunks', 'bathing trunks'],
     'Skirt':['miniskirt', 'mini']
      }
    have_glasses = {
        'Glasses': ['glasses', 'sunglass', 'sunglasses', 'dark glasses','shades']
        }
    wear_necklace = {
        'Necklace': ['neck_brace','necklace']
        }
    resnet_model = resnet50.ResNet50(weights='imagenet')
    test_image_resnet = image.load_img(path, target_size = (224, 224))
    test_image_resnet = image.img_to_array(test_image_resnet)
    test_image_resnet = np.expand_dims(test_image_resnet, axis = 0)
    result_resnet = resnet_model.predict(test_image_resnet)
    label = decode_predictions(result_resnet)
    data=[]
    o=label[0]
    data.append(o[1:])
    
    
    '''
    for element in range(len(label[0])):
        for key in valid_classes:
            if(label[0][element][1] in valid_classes[key]):
                if(float(label[0][element][2]) >= 0.05):
                    data.append(str(key))
                    break
              

    for element in range(len(label[0])):
        for key in have_glasses:
            if(label[0][element][1] in have_glasses[key]):
                if(float(label[0][element][2]) >= 0.01):
                    data.append("Yes,Wear Glasses")
    '''
                    
    for element in range(len(label[0])):
        for key in wear_necklace:
            if(label[0][element][1] in wear_necklace[key]):
                if(float(label[0][element][2]) >= 0.01):
                    data.append(str(key))
    return data 

def fn_image(request):
    if request.method =="POST":
        test_image = request.FILES['vi']
        fs = FileSystemStorage()
        img_name=test_image.name
        fs.save(test_image.name,test_image)
        path=str('media/'+str(img_name))
        data=[]
        test_image = image.load_img(path, target_size = (64, 64))
        test_image = image.img_to_array(test_image)
        test_image = np.expand_dims(test_image, axis = 0)
        now = datetime.now()

        current_time = now.strftime("%H:%M:%S")

        gender_var=gender(path)

        colors=color_pred(test_image)
        
        patterns=pattern_pred(test_image)
        
        cloth=features(path)
        k=str(current_time)
        k=labels(time_stamp=current_time,gender=gender_var,pattern=patterns,color=colors,cloths=cloth)
        k.save()
        print(data)
    return render(request,'image.html',{'prediction':'data'})



def video(request):
    if request.method =="POST":
        image = request.FILES['vi']
        fs = FileSystemStorage()
        img_name=image.name
        fs.save(image.name,image)
        cap = cv2.VideoCapture(str('media/'+str(img_name)))
        frame_width = int( cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height =int( cap.get( cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('X','V','I','D')
        out = cv2.VideoWriter("one.avi", fourcc, 5.0, (1280,720))
        ret, frame1 = cap.read()
        ret, frame2 = cap.read()
        print(frame1.shape)
        while cap.isOpened():
            diff = cv2.absdiff(frame1, frame2)
            gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
            blur = cv2.GaussianBlur(gray, (5,5), 0)
            _, thresh = cv2.threshold(blur, 20, 255, cv2.THRESH_BINARY)
            dilated = cv2.dilate(thresh, None, iterations=3)
            contours, _ = cv2.findContours(dilated, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                (x, y, w, h) = cv2.boundingRect(contour)

                if cv2.contourArea(contour) < 900:
                    continue
                cv2.rectangle(frame1, (x, y), (x+w, y+h), (0, 255, 0), 2)
                cv2.putText(frame1, "Status: {}".format('Movement'), (10, 20), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 0, 255), 3)
            #cv2.drawContours(frame1, contours, -1, (0, 255, 0), 2)

            image = cv2.resize(frame1, (1280,720))
            out.write(image)
            cv2.imshow("feed", frame1)
            frame1 = frame2
            ret, frame2 = cap.read()

            if cv2.waitKey(40) == ord('q'):
                break

        cv2.destroyAllWindows()
        cap.release()
        out.release()
    return render(request,'video.html')
def index_views(request):
    return render(request,'index.html')
def login(request):
    return render(request,'login.html')
def signup(request):
    form = UserCreationForm()
    if request.method == 'POST':
        form = UserCreationForm(request.POST)
        print(form)
        if form.is_valid():
            form.save()
            user=form.cleaned_data.get('username')
            messages.success(request,"Account created for"+user)
            return redirect('login')
    context={'form':form}
    return render(request,'reg.html',context)

def add(request):
    user=request.POST['user']
    pas=request.POST['password']
    if user == "inco" and pas=="9876":
        return render(request,'index.html')
    else:
        return HttpResponse("<h1>qwer</h1>") 
