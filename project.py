import cv2
import random
import imutils
import os
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
pytesseract.pytesseract.tesseract_cmd = 'C:\\Program Files\\Tesseract-OCR\\tesseract.exe'


images_dir = "C:/Users/user/Downloads/VIP/[TDS3651] Project/new img"
cropped_plate = "C:/Users/user/Downloads/VIP/[TDS3651] Project/cropped"

# fetch current working directory
current_path = os.getcwd()
# folder is fetched where license plate images are located dynamically
folder = os.path.join(current_path, 'true img')  #"true img" is the folder name
# folder to save cropped license plate
dirName = 'true'
count = 1

for filename in os.listdir(folder):
    img = cv2.imread((os.path.join(folder, filename)))
    write_folder = os.path.join(current_path, dirName)

    # name of file which is to be written to chosed folder
    image_name_to_write = (os.path.join(write_folder,'Cropped_{}.png'.format(count)))

    # convert to Grayscale Image
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # noise removal with bilateral filter
    biFilter = cv2.bilateralFilter(gray, 13, 17, 17)

    # Canny edge detection
    canny = cv2.Canny(biFilter, 170, 200)

    # find contours based on Edges
    contours = cv2.findContours(canny.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # sort contours based on minimum area 30
    contours = imutils.grab_contours(contours)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:30]

    # create copy of original image to draw contours
    image_copy = img.copy()
    _ = cv2.drawContours(image_copy, contours, -1, (255,0,255),2)

    contour_with_license_plate = None
    license_plate = None
    x = None
    y = None
    w = None
    h = None
    counts=1

    # loop over contours to find the best possible approximate contour of license plate
    for c in contours:
        perimeter = cv2.arcLength(c, True)
        approx = cv2.approxPolyDP(c, 0.018 * perimeter, True)
        if len(approx) == 4:
            contour_with_license_plate = approx
            x, y, w, h = cv2.boundingRect(c)
            img = cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            license_plate = gray[y:y + h, x:x + w]
            break

    if license_plate is None:
        detected = 0
        print("No contour detected!")
    else:
        detected = 1

    if detected == 1:
        image_copy = img.copy()
        cv2.drawContours(image_copy, contours, -1, (255,0,255),2)
        mask = np.zeros(gray.shape,np.uint8)
        cv2.drawContours(mask,[contour_with_license_plate],0,255,-1)

        (x, y) = np.where(mask == 255)
        (top_x, top_y) = (np.min(x), np.min(y))
        (bottom_x, bottom_y) = (np.max(x), np.max(y))
        crop = img[top_x:bottom_x+1, top_y:bottom_y+1]
        crop2 = gray[top_x:bottom_x+1, top_y:bottom_y+1]

    # image file is written using cv2.imwrite function
    write_images = cv2.imwrite(image_name_to_write, crop2)
    count = count+1

    text = pytesseract.image_to_string(crop2, config='--psm 8')
    #print("License plate is:", text) 
