import cv2
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'
import warnings 
import matplotlib.gridspec as gridspec
warnings.filterwarnings(action= 'ignore')
from os import listdir
import matplotlib.cm as cm
from os.path import isfile, join, splitext
import tensorflow as tf
from sklearn.metrics import f1_score 
from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, MaxPooling2D, Dropout, Conv2D, Activation
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import tensorflow.keras.backend as K
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
import os
from skimage.transform import resize
from skimage.io import imread

###############################################################################################################################################
input_dir = "Test2"
# output_dir = "output"
numImages = 76

onlyfiles = [f for f in listdir(input_dir) if isfile(join(input_dir, f))]
file = onlyfiles[0:numImages]
files =  sorted(file,key=lambda x: int(x.split('Cropped_')[1].split('.png')[0]))
count = 0

###############################################################################################################################################
for i, name in enumerate(files):
    img = cv2.imread(input_dir+ '/' + name, 0)
    plate_image = cv2.convertScaleAbs(img, alpha=(255.0))
    # binary = cv2.threshold(img, 180, 255,cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    binary = cv2.threshold(img,180,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY)[1]

    def sort_contours(cnts,reverse = False):
        i = 0
        boundingBoxes = [cv2.boundingRect(c) for c in cnts]
        (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),key=lambda b: b[1][i], reverse=reverse))
        return cnts
    cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # creat a copy version "test_roi" of plat_image to draw bounding box
    test_roi = plate_image.copy()
    # Initialize a list which will be used to append charater image
    crop_characters = []
    # define standard width and height of character
    digit_w, digit_h = 30, 60
    for c in sort_contours(cont):
        (x, y, w, h) = cv2.boundingRect(c)
        ratio = h/w
        if 1<=ratio<=3.5: # Only select contour with defined ratio
            if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                # Draw bounding box arroung digit number
                cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)
                # Sperate number and gibe prediction
                curr_num = binary[y:y+h,x:x+w]
                curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                crop_characters.append(curr_num)

###############################################################################################################################################

    Categories=['0', '1', '2', '3', '4', '5',
                '6', '7', '8', '9', 'A', 'B',
                'C', 'D', 'E', 'F', 'G', 'H',
                'I', 'J', 'K', 'L', 'M', 'N',
                'O', 'P', 'Q', 'R', 'S', 'T',
                'U', 'V', 'W', 'X', 'Y', 'Z']
    # flat_data_arr=[] #input array
    # target_arr=[] #output array
    
###############################################################################################################################################    
    print("Malaysia License Data")
    def load_keras_model(model_name):
    # Load json and create model
        json_file = open('./{}.json'.format(model_name), 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        model = model_from_json(loaded_model_json)
        # Load weights into new model
        model.load_weights("./{}.h5".format(model_name))
        return model      
    # store_keras_model(model, 'model_License_Plate')
    pre_trained_model = load_keras_model('model_Malaysia_Plate')
    model = pre_trained_model   
    output = []

###############################################################################################################################################    
    def fix_dimension(img):
        new_img = np.zeros((28,28,3))
        for i in range(3):
            new_img[:,:,i] = img 
        return new_img

    for i,ch in enumerate(crop_characters): #iterating over the characters
        img_ = cv2.resize(ch, (28,28), interpolation=cv2.INTER_AREA)
        img = fix_dimension(img_)
        img = img.reshape(1,28,28,3) #preparing image for the model
        y_ = model.predict_classes(img)[0] #predicting the class
        character = Categories[y_] #
        output.append(character) #storing the result in a list

    plate_number = ''.join(output)
    print(plate_number, file=open("ANN_output_MY.csv", "a"))
###############################################################################################################################################    
    # error = np.zeros((numImages,6),)
    # precision = np.zeros((numImages,6),)
    # recall = np.zeros((numImages,6),)
    # iou = np.zeros((numImages,6),)
    
    # for p in range(6):
    #     outputPart = (plate_number == p)*1
    #     gtPart = (gt == p)*1       
    #     precision[i,p] = sum(sum(gtPart*outputPart))/(sum(sum(outputPart))+eps)
    #     recall[i,p] = sum(sum(gtPart*outputPart))/sum(sum(gtPart))
    #     error[i,p] = 1 - ((2*precision[i,p]*recall[i,p])/(precision[i,p]+recall[i,p]+eps))
    #     iou[i,p] = sum(sum(gtPart*outputPart))/sum(sum(np.clip(gtPart+outputPart,0,1)))

        
    # avg_error = np.mean(error,axis=1)
    # avg_precision = np.mean(precision,axis=1)
    # avg_recall = np.mean(recall,axis=1)
    # avg_iou = np.mean(iou,axis=1)
    
    # print('Adapted Rand Error: %d%%' % (np.sum(avg_error)/numImages*100))
    # print('Precision: %d%%' % (np.sum(avg_precision)/numImages*100))
    # print('Recall: %d%%' % (np.sum(avg_recall)/numImages*100))
    # print('IoU: %d%%' % (np.sum(avg_iou)/numImages*100))