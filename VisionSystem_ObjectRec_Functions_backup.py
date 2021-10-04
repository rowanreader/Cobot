# from openpyxl import load_workbook
# from openpyxl import Workbook
# import pyrealsense2 as rs
import numpy as np
import cv2
import math
import time
import socket
import binascii
import random
# from Transform_code import rigid_transform_3D
# from Transform_code import yawpitchrolldecomposition
from PIL import Image

def find_cores3(cv_image):
    #finds_pixel_location_of_cores
    original = np.copy(cv_image)
    hsv_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    H,S,V = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    V = V * 2

    hsv_image = cv2.merge([H,S,V])
    image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    #Dx = cv2.Sobel(image,cv2.CV_8UC1,1,0)
    #Dy = cv2.Sobel(image,cv2.CV_8UC1,0,1)
    #M = cv2.addWeighted(Dx, 1, Dy,1,0)
    ret, binary = cv2.threshold(cv_image, 190, 255, cv2.THRESH_BINARY_INV)
    binary = binary.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    edges = cv2.Canny(binary, 50, 100)
    lines = []
    try:
      lines = cv2.HoughLinesP(edges,1,3.14/180,50,20,10)[0]
    except TypeError as e:
      print(e) 
    output = np.zeros_like(cv_image, dtype=np.uint8)

    count = 0
    count1 = 0
    listOslopesALengths = []
    CoreLines = []
    for line in lines:
      a = np.array((line[0] ,line[1]))
      b = np.array((line[2], line[3]))
      dist = np.linalg.norm(a-b)
      listOslopesALengths.append([(line[2]-line[0])/(line[3]-line[1]),dist])
    ListforSearch = listOslopesALengths
    for i in listOslopesALengths:
      for j in ListforSearch:
         try:
           x1 = float(i[0]+1.0)/float(j[0]+1.000001)
           y1 = float(i[1]+1.0)/float(j[1]+1.000001)
           hue, sat, V1 = hsv_image[lines[count][1], lines[count][0], 0], hsv_image[lines[count][1], lines[count][0], 1],hsv_image[lines[count][1], lines[count][0], 2]
           #print("hue", hue)
           #print("sat", sat)
           #y1 = float(i[1])/float(j[1])
           if (((x1)>=.9) and ((x1)<=1.1)) and ((y1>=.9) and (y1<=1.1)) :# and ((y1)>=.8) :# and ((y1)<=1.2))) and (float(i[1])>35.0)):
             CoreLines.append([lines[count],lines[count1]])
           #hue, sat, V1 = hsv_image[lines[count][1], lines[count][0], 0], hsv_image[lines[count][1], lines[count][0], 1],hsv_image[lines[count][1], lines[count][0], 2]
           #print("hue", hue)
           #print("sat", sat)
           if ((sat <= 40)):
             print("hue", hue)
             print("sat", sat)
             print("i",i)
             
         except ZeroDivisionError as e:
           print(e) 
         count1+=1
      count1 = 0
      count+=1
    if len(CoreLines) != 0:
      rect = []
      for line in CoreLines:
        cv2.line(cv_image,(line[0][0],line[0][1]), (line[0][2], line[0][3]), (100,200,50), 2, cv2.LINE_AA)
        cv2.line(cv_image,(line[1][0],line[1][1]), (line[1][2], line[1][3]), (100,200,50), 2, cv2.LINE_AA)
        rect.append([line[0][0],line[0][1], line[1][2]-line[0][0], line[1][3]-line[0][1]])

      #points = np.array([np.transpose(np.where(output != 0))], dtype=np.float32)
      #rect = cv2.boundingRect(points)

    retval, threshold = cv2.threshold(cv_image, 190, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
 
    ret,thresh = cv2.threshold(gray,127,255,1)
    
    """image, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for cnt in contours:
      approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
      #print len(approx)
      if len(approx)==5:
        print ("pentagon")
        #cv2.drawContours(cv_image,[cnt],0,255,-1)
      elif len(approx)==3:
        print ("triangle")
        #cv2.drawContours(cv_image,[cnt],0,(0,255,0),-1)
      elif len(approx)==4:
        (x, y, w, h) = cv2.boundingRect(approx)
	ar = w / float(h)
        # a square will have an aspect ratio that is approximately
	# equal to one, otherwise, the shape is a rectangle
	shape = "square" if ar >= 0.95 and ar <= 1.05 else "rectangle"
        print ("square")
        #cv2.rectangle((cv_image),(x,y),(x+w,y-h),(255,0,0),5)
        #cv2.drawContours(cv_image,[cnt],0,(0,0,255),5)
      elif len(approx) == 9:
        print ("half-circle")
        cv2.drawContours(cv_image,[cnt],0,(255,255,0),-1)
      elif len(approx) > 15:
        print ("circle")
        #cv2.drawContours(cv_image,[cnt],0,(0,255,255),-1)"""
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10;
    params.maxThreshold = 250;

    blur = cv2.GaussianBlur(cv_image,(5,5),0)
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 100000
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # Set up the detector with default parameters."""
    #detector = cv2.SimpleBlobDetector()

    # Detect blobs.
    keypoints = detector.detect(threshold)

    # Draw detected blobs as red circles.
    # cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS ensures the size of the circle corresponds to the size of blob
    try:
      for stuff in rect:
        cv2.rectangle(cv_image,(stuff[0],stuff[1]), (stuff[0]+stuff[2], stuff[1]+stuff[3]),(255,255,255),thickness=2)
    except UnboundLocalError as e:
      print(e)
    #original = cv2.cvtColor(original,cv2.COLOR_BGR2RGB)
    im_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    return(im_with_keypoints)


def find_cores1(cv_image):
    ret, binary = cv2.threshold(cv_image, 190, 255, cv2.THRESH_BINARY_INV)
    #mask = np.zeros(self.edge.shape, np.uint8)
    binary = binary.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    #binary = cv2.morphologyEx(cv2.bitwise_and(self.edge, self.edge, mask = mask), cv2.MORPH_CLOSE, np.array([[1] * EDGE_DILATION] *EDGE_DILATION))
    binary1 = cv2.Canny(binary, 50, 100)
    kernel = np.ones((5,5), np.uint8)
    #img_erosion = cv2.erode(img, kernel, iterations=1) 
    #binary1 = cv2.dilate(binary1, kernel, iterations=5) 
    #binary1 = cv2.erode(binary1, kernel, iterations=3) 
    #binary1 = cv2.cvtColor(binary1, cv2.COLOR_BGR2GRAY)
    #ret,binary1 = cv2.threshold(binary1,127,255,cv2.THRESH_BINARY_INV)
    box = []
    #cv2.imshow("aruco", binary1)
    #cv2.waitKey(2)
    contours,hierarchy = cv2.findContours(binary1, 1, 2)
    for cnt in contours: 
      #cnt = contours[1]
      area = cv2.contourArea(cnt)
      if (area > 1000.0): #and (area != self.area):
        
        #print(area)
        #M = cv2.moments(cnt)
        perimeter = cv2.arcLength(cnt,True)
        epsilon = 0.1*cv2.arcLength(cnt,True)
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        hull = cv2.convexHull(cnt)
        rect = cv2.minAreaRect(cnt)
        rect1 = cv2.minAreaRect(approx)
        box11 = cv2.boxPoints(rect1)
        boxe = cv2.boxPoints(rect)
        box11 = np.int0(box11)
        print("box", boxe[0][0])
        print("box", boxe[0][1])
        
        boxe = np.int0(boxe)
        print("np_box", boxe)
        
        #a = np.array((line[0] ,line[1]))
        #b = np.array((line[2], line[3]))
        dist = np.linalg.norm(boxe[0]-boxe[1])
        dist1 = np.linalg.norm(boxe[1]-boxe[2])
        dist2 = np.linalg.norm(boxe[1]-boxe[3])
        distlist = [dist,dist1,dist2]
        distlist.sort()
        print("dist", dist)
        print("dist1", dist1)
        print("dist2", dist2)
        
        #if (((dist/dist1) > 3.7) and ((dist/dist1)<4.3) and (area > 35)) :
        #if ((dist1/dist) > 3.3) and (dist1*dist > 5500): #and (dist1*dist < 8000):
        #cv2.drawContours(cv_image,[box],0,(0,0,255),2)
        #print("distlist", distlist[0])
        
        if ((float(distlist[0])*distlist[1]) > 3000)  and ((distlist[1]/distlist[0]) > 2.7): #and (distlist[0]*distlist[1] < 10000):
          print("AREA",area)
          
        cv2.drawContours(cv_image,[boxe],0,(0,0,255),2)
        cv2.drawContours(cv_image,[box11],0,(0,255,0),2)
        cv2.drawContours(binary1,[boxe],0,(0,0,255),2)
        cv2.drawContours(binary1,[box11],0,(0,255,0),2)
        return(cv_image)
          #cv2.imshow("aruco", binary1)

def find_cores(cv_image):
    original = np.copy(cv_image) # make copy of image that won't be mutated
    #hsv_image = cv2.cvtColor(original, cv2.COLOR_BGR2HSV)
    #H,S,V = hsv_image[:,:,0], hsv_image[:,:,1], hsv_image[:,:,2]
    #V = V * 2

    #hsv_image = cv2.merge([H,S,V])
    #image = cv2.cvtColor(hsv_image, cv2.COLOR_HSV2RGB)
    image = cv2.cvtColor(cv_image, cv2.cv2.COLOR_BGR2GRAY) # convert to grayscale, appme BGR initially
    #Dx = cv2.Sobel(image,cv2.CV_8UC1,1,0)
    #Dy = cv2.Sobel(image,cv2.CV_8UC1,0,1)
    #M = cv2.addWeighted(Dx, 1, Dy,1,0)

    # threshold, separates from background, becomes either 0 of 255
    # returns threshold value and modified thesholded image
    ret, binary = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY_INV)
    # convert to int8
    binary = binary.astype(np.uint8)
    # finds ellipse, applies closing - connects close elements
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (20, 20)))
    # finds edges using canny edge detection
    edges = cv2.Canny(binary, 50, 100)
    lines = []    
    box = []
    #cv2.imshow("aruco", binary1)
    #cv2.waitKey(2)
    #cv2.cvtColor(binary, cv2.COLOR_BGR2GRAY);
    # finds all boundary points of an image
    # returns contours, a list of all x,y coords of boundary
    # and hierarchy, the parent-child relationship [next contour, prev contour, 1st child contour, parent contour]
    # because we use RETR_LIST, doesn't make parent-child relationships
    # also, doesn't return every point - if its a line it just returns the ends of the line
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    found_pixels = []
    theta = []
    for cnt in contours: 
      #cnt = contours[1]
      area = cv2.contourArea(cnt) # finds area of contour
      # check if area is within range
      if (area > 2000.0) and (area < 3000.0) : #and (area != self.area):
        
        #print(area)
        #M = cv2.moments(cnt)
        # finds perimiter of contour assuming it is closed
        perimeter = cv2.arcLength(cnt,True)
        epsilon = 0.1*cv2.arcLength(cnt,True)

        # approximates the shape of the contour
        approx = cv2.approxPolyDP(cnt,epsilon,True)
        # makes an approximation of the overall shape
        hull = cv2.convexHull(cnt)
        # find minimal rectangle that encompases
        rect = cv2.minAreaRect(cnt)
        rect1 = cv2.minAreaRect(approx)
        # get 4 corners of the rectangle
        box11 = cv2.boxPoints(rect1)
        width = int(rect[1][0])
        height = int(rect[1][1])
        #print(rect.size.width)
        # finds orientation?
        if (width < height): 
            theta.append(rect[2] - 90)
        else:
            theta.append(rect[2])

        boxe = cv2.boxPoints(rect)
        box_d = np.int0(box11)
        box_f = np.int0(boxe) # converts from float to int
        #mask 
        print("box", boxe[0][0])
        print("box", boxe[0][1])
        found_pixels.append(boxe)
        
        #cv2.drawContours(cv_image,[box_d],0,(0,0,255),2)
        # draws contours on the image
        cv2.drawContours(cv_image,[box_f],0,(0,255,0),2)
    #cv2.drawContours(binary1,[boxe],0,(0,0,255),2)
    #cv2.drawContours(binary1,[box11],0,(0,255,0),2)
        print(cv_image.shape)
    # returns image (modified?), pixels of surrounding rectangle, and angle of rotation
    return(cv_image, found_pixels, theta)


def getRotation(coords): 
    """
    *Given coordinages of [x1, y1, x2, y2, x3, y3, x4, y4]
    *  where the corners are:
    *            top left    : x1, y1
    *            top right   : x2, y2
    *            bottom right: x3, y3
    *            bottom left : x4, y4"""
    # Get center as average of top left and bottom right
    center = [(coords[0][0] + coords[1][0] + coords[2][0]+ coords[3][0]) / 4.0,
                  (coords[0][1] + coords[1][1] + coords[2][1]+ coords[3][1] ) / 4.0]

    #Get differences top left minus bottom left
    diffs = [coords[0][0] - coords[3][0], coords[0][1] - coords[3][1]]

    #Get rotation in degrees
    rotation = math.atan2(diffs[0] , diffs[1]) * 180 / math.pi

    # Adjust for 2nd & 3rd quadrants, i.e. diff y is -ve.
    if (diffs[0] > diffs[1]):
        rotation += 90

    # Adjust for 4th quadrant
    # i.e. diff x is -ve, diff y is +ve
    elif (diffs[0] < 0):
        rotation += 360
    
    # return array of [[centerX, centerY], rotation];
    outlist = [center[0], center[1], rotation]
    print("rotation", rotation)
    return (rotation-290)


def find_tops(cv_image):
    original = np.copy(cv_image)
    retval, threshold = cv2.threshold(cv_image, 190, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
 
    ret,thresh = cv2.threshold(gray,127,255,1)
    # blob detection
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10; # set params for blob detection
    params.maxThreshold = 250;

    blur = cv2.GaussianBlur(cv_image,(5,5),0)
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 10000
    # get cv2 version
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)

    # actually detect
    keypoints = detector.detect(threshold)
    try:
        x = keypoints[0].pt[0] #i is the index of the blob you want to get the position
        y = keypoints[0].pt[1]
        diameter = keypoints[0].size
        im_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        success = 1
    except:
        x = 0.0
        y = 0.0
        diameter = 0.0
        im_with_keypoints = cv_image
        success = 0
    #image = cv2.cvtColor(cv_image, cv2.cv2.COLOR_BGR2GRAY)
    
    return(im_with_keypoints, x, y, diameter, success)

# same as find_tops????
def find_tops2(cv_image):
    
    original = np.copy(cv_image)
    retval, threshold = cv2.threshold(cv_image, 190, 255, cv2.THRESH_BINARY_INV)
    gray = cv2.cvtColor(threshold, cv2.COLOR_BGR2GRAY)
 
    ret,thresh = cv2.threshold(gray,127,255,1)
    params = cv2.SimpleBlobDetector_Params()
    params.minThreshold = 10;
    params.maxThreshold = 250;

    blur = cv2.GaussianBlur(cv_image,(5,5),0)
    params.filterByColor = True
    params.blobColor = 255
    params.filterByCircularity = True
    params.minCircularity = 0.5
    params.filterByConvexity = True
    params.minConvexity = 0.87
    params.filterByArea = True
    params.minArea = 500
    params.maxArea = 10000
    ver = (cv2.__version__).split('.')
    if int(ver[0]) < 3 :
        detector = cv2.SimpleBlobDetector(params)
    else :
        detector = cv2.SimpleBlobDetector_create(params)
    keypoints = detector.detect(threshold)
    try:
        x = keypoints[0].pt[0] #i is the index of the blob you want to get the position
        y = keypoints[0].pt[1]
        diameter = keypoints[0].size
        im_with_keypoints = cv2.drawKeypoints(cv_image, keypoints, np.array([]), (0,0,255), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
        success = 1
    except:
        x = 0.0
        y = 0.0
        diameter = 0.0
        im_with_keypoints = cv_image
        success = 0
    #image = cv2.cvtColor(cv_image, cv2.cv2.COLOR_BGR2GRAY)
    
    return(im_with_keypoints, x, y, diameter, success)

img = cv2.imread("singleRedPillar1.jpg")
newIm, pxl, theta = find_cores(img)
newIm = find_cores1(img)
newIm = find_cores3(img)
cv2.imshow('Image', newIm)
cv2.waitKey(0)