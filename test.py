import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import pickle
import pyttsx3

def main():
    input_img = np.full((480, 640), 0, dtype=np.uint8)

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC,60*1000)

    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils

    prev=0
    points=[]
    middle_y=index_y=0

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks
        cv2.namedWindow("Resized_Window", cv2.WINDOW_NORMAL)

        cv2.resizeWindow("Resized_Window", 480, 640)

        if hands:
            cur=len(hands)
            if (middle_y<index_y):
                if (np.count_nonzero(input_img) > 3400):
                    input_img_resized = cv2.resize(input_img, (28, 28))
                    input_img_reshaped=input_img_resized.reshape((1, 28, 28, 1))
                    value1= pickled_model.predict(input_img_reshaped)
                    value=value1.argmax()
                    print(value)
                    value=int(value)
                    valuechar=chr(value)
                    if(value>=10 and value<=36):
                        valuechar=chr(value+55)
                    points.clear()
                    engine=pyttsx3.init()
                    engine.setProperty('rate', 100)
                    text=valuechar
                    engine.say(text)
                    engine.runAndWait()
                    plt.imshow(input_img_resized)
                    plt.show()
                input_img = np.full((480, 640), 0, dtype=np.uint8)

            for hand in hands:
                drawing_utils.draw_landmarks(rgb_frame, hand) 
                landmarks = hand.landmark

                for id, landmark in enumerate(landmarks):
                    x = int((landmark.x) * frame_width)
                    y = int((landmark.y) * frame_height)

                    if id == 8:
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        index_y=y
                        points.append([x, y])
                        if (len(points) == 2):
                            cv2.line(input_img, points[0], points[1], color=(255, 255, 255), thickness=30)
                            points.pop(0)
                    if id==12:
                        middle_y=y
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
            prev=cur

        cv2.imshow("img", input_img)
        cv2.imshow("PBL Gesture", rgb_frame) 
        
        if cv2.waitKey(1) & 0xFF==27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    pickled_model = keras.models.load_model('planB/my_model.h5')
    main()
