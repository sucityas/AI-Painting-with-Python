import cv2
import numpy as np
import time
import os 
import os 
import HandTrackingModule as htm

detector = htm.handDetector(detectionConfidence = 0.85) 

cap = cv2.VideoCapture(0)
cap.set(3, 1280) # set the canvas width
cap.set(4, 720) # set the canvas height
myListDirectory = os.listdir("virtual_painter/header") #read filel inside folder header
print(myListDirectory)
overlayList = []

for imPath in myListDirectory:
    image = cv2.imread(f'virtual_painter/header/{imPath}')
    overlayList.append(image)

header = overlayList[0] #index ke-0 for overlayList as a default header
drawColor = (0,0,255) # default red brush
brushThickness = 7    # brush thickness
eraserThickness = 40  # eraser thickness
xp, yp = 0,0          # start coordinate
imgCanvas = np.zeros((720, 1280, 3), np.uint8) 

while True:
    res, frame = cap.read()
    frame = cv2.flip(frame, 1)  # flip frames
    frame = detector.findHands(frame) #called findHands() method
    lmList = detector.findPosition(frame, draw=True)
    if len(lmList) != 0:
        x1, y1 = lmList[8][1:]
        x2, y2 = lmList[12][1:]
        print(x1, y1, x2, y2)
        fingers = detector.fingersUp()
        if fingers[1] and fingers[2]: 
            xp, yp = 0,0 
            print("Selection mode") 
            cv2.rectangle(frame, (x1,y1-25), (x2, y2+25), drawColor, cv2.FILLED)
        if y1<125:
            if 320 < x1 < 480:
                header = overlayList[0]
                drawColor = (0,0,255)
            elif 480 < x1 < 630:
                header = overlayList[1]
                drawColor = (0, 255, 0)
            elif 630 < x1 < 840:
                header = overlayList[2]
                drawColor = (255,0,0)
            elif x1> 1000:
                header = overlayList[3]
                drawColor = (0,0,0)  # For eraser mode, brush will be black color
        if fingers[1] and fingers[2] == False:
            print("Drawing Mode")
            cv2.circle(frame, (x1, y1), 15, drawColor, cv2.FILLED)
            if xp == 0 and yp == 0: 
                xp, yp = x1, y1
            if drawColor == (0,0,0):                                             
                cv2.line(frame, (xp, yp), (x1,y1), drawColor,eraserThickness)   
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor,eraserThickness) 
            else:                                  
                cv2.line(frame, (xp, yp), (x1,y1), drawColor,brushThickness)  
                cv2.line(imgCanvas, (xp, yp), (x1,y1), drawColor,brushThickness) #Drawing in black canvas
            xp, yp = x1, y1
    frameGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
    _, frameInvers = cv2.threshold(frameGray, 50, 255, cv2.THRESH_BINARY_INV)
    frameInvers = cv2.cvtColor(frameInvers, cv2.COLOR_GRAY2BGR)
    frame = cv2.bitwise_and(frame, frameInvers)
    frame = cv2.bitwise_or(frame, imgCanvas)
    frame[0: 125, 0:1280] = header
    cv2.imshow("Canvas", imgCanvas)
    cv2.imshow("Frame", frame) 
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()