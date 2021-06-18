
import cv2
import numpy as np


#STACK IMAGES-------------------------------------------------
def stackImages(scale,imgArray):
    rows = len(imgArray)
    cols = len(imgArray[0])
    rowsAvailable = isinstance(imgArray[0], list)
    width = imgArray[0][0].shape[1]
    height = imgArray[0][0].shape[0]
    if rowsAvailable:
        for x in range(0, rows):
            for y in range(0, cols):
                if imgArray[x][y].shape[:2] == imgArray[0][0].shape[:2]:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (0, 0), None, scale, scale)
                else:
                    imgArray[x][y] = cv2.resize(imgArray[x][y], (imgArray[0][0].shape[1], imgArray[0][0].shape[0]), None, scale, scale)
                if len(imgArray[x][y].shape) == 2: imgArray[x][y]= cv2.cvtColor( imgArray[x][y], cv2.COLOR_GRAY2BGR)
        imageBlank = np.zeros((height, width, 3), np.uint8)
        hor = [imageBlank]*rows
        hor_con = [imageBlank]*rows
        for x in range(0, rows):
            hor[x] = np.hstack(imgArray[x])
        ver = np.vstack(hor)
    else:
        for x in range(0, rows):
            if imgArray[x].shape[:2] == imgArray[0].shape[:2]:
                imgArray[x] = cv2.resize(imgArray[x], (0, 0), None, scale, scale)
            else:
                imgArray[x] = cv2.resize(imgArray[x], (imgArray[0].shape[1], imgArray[0].shape[0]), None,scale, scale)
            if len(imgArray[x].shape) == 2: imgArray[x] = cv2.cvtColor(imgArray[x], cv2.COLOR_GRAY2BGR)
        hor = np.hstack(imgArray)
        ver = hor
    return ver


#IMAGE-------------------------------------------------
img = cv2.imread("Resources/earth.png")
cv2.imshow("Output",img)
cv2.waitKey(0)


#WEBCAM-------------------------------------------------
cap = cv2.VideoCapture(0) # 0 shows webcam
cap.set(3,640) #width
cap.set(4,480) #height
cap.set(10,100) #brightness

while True:
    success, img = cap.read()
    cv2.imshow("Vid", img)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break


#GRAY-------------------------------------------------
img = cv2.imread("Resources/earth.png", 0) # 0 means grayscale
cv2.imshow("Output",img)
cv2.waitKey(0)


#BLUR-------------------------------------------------
img = cv2.imread("Resources/earth.png")
imgblur = cv2.GaussianBlur(img,(27,27),0) # ratio odd numbers only
cv2.imshow("Output",imgblur)
cv2.waitKey(0)


#EDGE KALINLIK INCELIK-------------------------------------------------
img = cv2.imread("Resources/earth.png")
kernel = np.ones((5,5), np.uint8)

imgcanny = cv2.Canny(img,150,200)
imgdilation = cv2.dilate(imgcanny,kernel,iterations=1)
imgeroded = cv2.erode(imgdilation,kernel,iterations=1)

cv2.imshow("Output1",imgcanny)
cv2.imshow("Output2",imgdilation)
cv2.imshow("Output3",imgeroded)
cv2.waitKey(0)


#SIZE CROP-------------------------------------------------
img = cv2.imread("Resources/earth.png")
imgcut = cv2.imread("Resources/mac.png")

width, height = 250,300
pts1 = np.float32([[90,61],[189,4],[151,176],[263,114]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgfinal = cv2.warpPerspective(imgcut,matrix,(width,height))

imgcropped = img[500:800,500:800]
cv2.imshow("Output",imgcropped)
cv2.imshow("Img Cut",imgfinal)
cv2.waitKey(0)


#SHAPES TEXTS-------------------------------------------------
img = np.zeros((512,512,3),np.uint8)

img[:] = 150,60,60
img[200:400,100:400] = 0,0,255

cv2.line(img,(10,10),(500,500),(0,255,0),2)
cv2.rectangle(img,(200,10),(400,100),(150,50,20),cv2.FILLED)
cv2.circle(img,(100,400),50,(10,10,10),cv2.FILLED)
cv2.putText(img,"KUZEY",(150,300),cv2.FONT_HERSHEY_COMPLEX,2,(255,180,0),3)

cv2.imshow("Output",img)
cv2.waitKey(0)


#JOIN IMAGES-------------------------------------------------
earth = cv2.imread("Resources/earth.png")
mars = cv2.imread("Resources/mars.png")
moon = cv2.imread("Resources/moon.png")

# marsres = cv2.resize(mars,(1789, 1787))
# moonres = cv2.resize(moon,(1789, 1787))

# imghor = np.hstack((earth,moonres,marsres))
# imgver = np.vstack((earth,moonres,marsres))

imgstacked = stackImages(0.5,([moon,mars],[earth,moon]))

cv2.imshow("Output", imgstacked)
cv2.waitKey(0)


#COLOR DETECTION-------------------------------------------------
def empty(a):
    pass

cv2.namedWindow("TrackBars")
cv2.resizeWindow("TrackBars",1640,1240)
cv2.createTrackbar("Hue Min","TrackBars",12,179,empty)
cv2.createTrackbar("Hue Max","TrackBars",26,179,empty)
cv2.createTrackbar("Sat Min","TrackBars",76,255,empty)
cv2.createTrackbar("Sat Max","TrackBars",254,255,empty)
cv2.createTrackbar("Val Min","TrackBars",160,255,empty)
cv2.createTrackbar("Val Max","TrackBars",255,255,empty)

while True:
    path = 'Resources/lambo.jpg'
    img = cv2.imread(path)
    imgres = cv2.resize(img, (450, 300))
    imghsv = cv2.cvtColor(img,cv2.COLOR_BGR2HSV)
    imghsvres = cv2.resize(imghsv, (450, 300))
    imgblack = np.zeros_like(imgres)
    imgcanny = cv2.Canny(imgres,450,450)

    h_min = cv2.getTrackbarPos("Hue Min","TrackBars")
    h_max = cv2.getTrackbarPos("Hue Max","TrackBars")
    s_min = cv2.getTrackbarPos("Sat Min","TrackBars")
    s_max = cv2.getTrackbarPos("Sat Max","TrackBars")
    v_min = cv2.getTrackbarPos("Val Min","TrackBars")
    v_max = cv2.getTrackbarPos("Val Max","TrackBars")
    print(h_min,h_max,s_min,s_max,v_min,v_max)

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imghsvres,lower,upper)

    contours, hierarchy = cv2.findContours(imgcanny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    cv2.drawContours(imgblack, contours, -1,(255,0,0),1)

    imgresult = cv2.bitwise_and(imgres,imgres,mask=mask)
    stacked = stackImages(1,([imgres,imghsvres,imgcanny],[mask,imgresult,imgblack]))

#    cv2.imshow("Original", imgres)
#    cv2.imshow("HSV", imghsvres)
#    cv2.imshow("HSVmasked", mask)
#    cv2.imshow("RESULT", imgresult)
    cv2.imshow("Finished Product", stacked)
    cv2.waitKey(1)


#SHAPE DETECTION-------------------------------------------------
img = cv2.imread("Resources/shapes.png")
imgcopy = img.copy()
imggray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
imgblur = cv2.GaussianBlur(imggray,(3,3),1)
imgblack = np.zeros_like(img)
imgcanny = cv2.Canny(imgblur,50,50)

def getContours(img):
    contours, hierarchy = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    for cnt in contours:
        area = cv2.contourArea(cnt)
        print(area)
        if area > 500:
            cv2.drawContours(imgcopy, cnt, -1,(255,0,0),2)
            peri = cv2.arcLength(cnt, True)
            print((peri))
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            print(len(approx))
            corner = len(approx)
            x, y, w, h = cv2.boundingRect(approx)

            if corner == 3:
                objectType = "Triangle"
            elif corner == 4:
                if 0.95 < w/float(h) < 1.05:
                    objectType = "Square"
                else:
                    objectType = "Rectangle"
            elif corner > 4:
                objectType = "Circle"

            cv2.rectangle(imgcopy,(x,y),(x+w, y+h),(0,255,0),2)
            cv2.putText(imgcopy, objectType,(x+(w//2)-10,y+(h//2)),cv2.FONT_HERSHEY_COMPLEX,0.7,(0,0,0),2)


getContours(imgcanny)

imgstacked = stackImages(0.8,([img,imggray,imgblur],[imgcanny,imgcopy,imgblack]))
cv2.imshow("Shapes", imgstacked)
cv2.waitKey(0)


#FACE DETECTION-------------------------------------------------
img = cv2.imread("Resources/mac.png")
imgcut = cv2.imread("Resources/mac.png")

width, height = 250,300
pts1 = np.float32([[90,61],[189,4],[151,176],[263,114]])
pts2 = np.float32([[0,0],[width,0],[0,height],[width,height]])
matrix = cv2.getPerspectiveTransform(pts1,pts2)
imgfinal = cv2.warpPerspective(imgcut,matrix,(width,height))

faceCascade = cv2.CascadeClassifier("Resources/haarcascade_frontalface_default.xml")

faces1 = faceCascade.detectMultiScale(img,1.1,2)
for (x, y, w, h) in faces1:
    cv2.rectangle(img,(x,y),(x+w, y+h),(0,255,0),2)

faces2 = faceCascade.detectMultiScale(imgfinal,1.1,2)
for (x, y, w, h) in faces2:
    cv2.rectangle(imgfinal,(x,y),(x+w, y+h),(0,255,0),2)

cv2.imshow("Result", img)
cv2.imshow("Img Cut", imgfinal)
cv2.waitKey(0)


#COLOR PICKER-------------------------------------------------
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)

def empty(a):
    pass

cv2.namedWindow("HSV")
cv2.resizeWindow("HSV",640,240)
cv2.createTrackbar("H min", "HSV", 0, 179, empty)
cv2.createTrackbar("H max", "HSV", 179, 179, empty)
cv2.createTrackbar("S min", "HSV", 0, 255, empty)
cv2.createTrackbar("S max", "HSV", 255, 255, empty)
cv2.createTrackbar("V min", "HSV", 0, 255, empty)
cv2.createTrackbar("V max", "HSV", 255, 255, empty)

while True:

    _, img = cap.read()
    imgflip = cv2.flip(img, 1)
    imghsv = cv2.cvtColor(imgflip, cv2.COLOR_BGR2HSV)

    h_min = cv2.getTrackbarPos("H min", "HSV")
    h_max = cv2.getTrackbarPos("H max", "HSV")
    s_min = cv2.getTrackbarPos("S min", "HSV")
    s_max = cv2.getTrackbarPos("S max", "HSV")
    v_min = cv2.getTrackbarPos("V min", "HSV")
    v_max = cv2.getTrackbarPos("V max", "HSV")

    lower = np.array([h_min,s_min,v_min])
    upper = np.array([h_max,s_max,v_max])
    mask = cv2.inRange(imghsv,lower,upper)
    result = cv2.bitwise_and(imgflip, imgflip, mask=mask)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    hstack = np.hstack([imgflip, mask, result])

    print([h_min,h_max,s_min,s_max,v_min,v_max])

    cv2.imshow("Color Picker", hstack)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#PROJECT 1-----VIRTUAL PAINT----------------------------------
cap = cv2.VideoCapture(0)
cap.set(3,640)
cap.set(4,480)
cap.set(10,150)

myColors = [[116,113,183,179,255,255], [27,227,80,100,255,221], [94,124,97,164,255,215]]
myColorValues = [[69,0,255],[71,100,0],[154,32,50]]
myPoints = []

def findColor(imgflip, myColors, myColorValues):
    imghsv = cv2.cvtColor(imgflip,cv2.COLOR_BGR2HSV)
    count = 0
    newPts = []
    for color in myColors:
        lower = np.array(color[0:3])
        upper = np.array(color[3:6])
        mask = cv2.inRange(imghsv,lower,upper)
        x,y = getContours(mask)
        cv2.circle(imgresult,(x,y),10,myColorValues[count],cv2.FILLED)
        if x != 0 and y != 0:
            newPts.append([x, y, count])
        count += 1
        cv2.imshow(str(color[0]), mask)
    return newPts

def getContours(imgflip):
    contours, hierarchy = cv2.findContours(imgflip, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    x,y,w,h = 0,0,0,0
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area > 500:
            #cv2.drawContours(imgresult, cnt, -1,(255,0,0),3)
            peri = cv2.arcLength(cnt, True)
            approx = cv2.approxPolyDP(cnt,0.02*peri, True)
            x, y, w, h = cv2.boundingRect(approx)
    return x + w//2, y

def draw(myPoints, myColorValues):
    for point in myPoints:
        cv2.circle(imgresult, (point[0], point[1]), 10, myColorValues[point[2]], cv2.FILLED)


while True:
    success, img = cap.read()
    imgflip = cv2.flip(img, 1)
    imgresult = imgflip.copy()
    newPts = findColor(imgflip, myColors, myColorValues)
    if len(newPts) != 0:
        for newP in newPts:
            myPoints.append(newP)
    if len(myPoints) != 0:
        draw(myPoints, myColorValues)

    cv2.imshow("Result", imgresult)
    if cv2.waitKey(1) & 0xFF ==ord('q'):
        break

        
