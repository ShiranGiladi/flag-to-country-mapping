import cv2
import mediapipe as mp
import time
import math

class handDetector():
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # thumb, index, middle, ring, pinkey fingers
        self.tipIds = [4, 8, 12, 16, 20]

    # draw the landmarks and the connections line on the img
    def findHands(self, img, draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)

        return img
    
    # return an array with the position of all the landmarks in a single frame
    # in case there is more then one hand on the screen, handNo pick a specific hand
    def findPosition(self, img, handNo=0, draw=True):

        self.lmList = []

        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]

            for id, lm in enumerate(myHand.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x*w), int(lm.y*h)
                self.lmList.append([id, cx, cy])
                if draw:
                    cv2.circle(img, (cx,cy), 10, (255,0,0), cv2.FILLED)
        
        return self.lmList
    
    # return an array that represent which fingers is up(1) / down(0)
    def fingersUp(self):
        fingers = []

        # thumb
        if self.lmList[self.tipIds[0]][1] < self.lmList[self.tipIds[0]-1][1]: # compare using the x axis
            fingers.append(1)
        else:
            fingers.append(0)

        # 4 fingers
        for id in range(1, 5):
            if self.lmList[self.tipIds[id]][2] < self.lmList[self.tipIds[id]-2][2]: # compare using the y axis
                fingers.append(1)
            else:
                fingers.append(0)

        return fingers
    
    def findDistance(self, p1, p2, img, draw=True, r=15, t=3):
        x1, y1 = self.lmList[p1][1:]
        x2, y2 = self.lmList[p2][1:]
        # center point
        cx, cy = (x1+x2)//2, (y1+y2)//2

        if draw:
            cv2.line(img, (x1,y1), (x2,y2), (255,0,255), t)
            cv2.circle(img, (x1,y1), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (x2,y2), r, (255,0,255), cv2.FILLED)
            cv2.circle(img, (cx,cy), r, (255,0,255), cv2.FILLED)

        # calculate the length of the line
        length = math.hypot(x2-x1, y2-y1)

        return length, img, [x1, y1, x2, y2, cx, cy]


def main():
    cap = cv2.VideoCapture(0)

    pTime = 0
    cTime = 0

    detector = handDetector()

    while True:
        success, img = cap.read()
        img = detector.findHands(img)

        # lmList contains all the positions of the landmarks, each id is represent a dot in the hand (there is 0-20 dots)
        lmList = detector.findPosition(img)
        if len(lmList) != 0:
            # here we print all the positions for the landmark 4 (thumb tip)
            print(lmList[4])

        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        cv2.putText(img, str(int(fps)), (10,70), cv2.FONT_HERSHEY_PLAIN, 3, (255,0,255), 3)
        cv2.imshow('Image', img)
        cv2.waitKey(1)


if __name__ == "__main__":
    main()