import cv1
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

def main():
    input_img = np.full((800, 800), 0, dtype=np.uint8)

    cap=cv2.VideoCapture(0)

    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils

    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            for hand in hands:
                drawing_utils.draw_landmarks(rgb_frame, hand) 
                landmarks = hand.landmark

                for id, landmark in enumerate(landmarks):
                    x = int((landmark.x) * frame_width)
                    y = int((landmark.y) * frame_height)

                    if id == 8:
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        cv2.circle(input_img, (x,y), radius=10, color=(255, 255, 255), thickness=-1)

        cv2.imshow("img", input_img)
        cv2.imshow("PBL Gesture", rgb_frame) 
        
        if cv2.waitKey(1) & 0xFF==ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
