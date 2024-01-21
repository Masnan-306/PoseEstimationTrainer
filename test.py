import cv2
from pdb import set_trace as bp
from PoseModule import PoseDetector
# from cvzone.PoseModule import PoseDetector
detector = PoseDetector()
cap=cv2.VideoCapture(0)

while True:
    bp()
    success,img=cap.read()
    img = detector.findPose(img)
    lmList,bbox = detector.findPosition(img)
    cv2.imshow("Result",img)
    cv2.waitKey(1)
