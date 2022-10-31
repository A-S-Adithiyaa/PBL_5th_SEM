import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
import numpy as np

def main():
    input_img = np.full((480, 640), 255, dtype=np.uint8)

    cap=cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_POS_MSEC,60*1000)

    hand_detector = mp.solutions.hands.Hands()
    drawing_utils = mp.solutions.drawing_utils

    prev=0
    points=[]
    while True:
        _, frame = cap.read()
        frame = cv2.flip(frame, 1)
        frame_height, frame_width, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        output = hand_detector.process(rgb_frame)
        hands = output.multi_hand_landmarks

        if hands:
            cur=len(hands)
            if (cur == 2 and prev == 2):
                if (np.count_nonzero(input_img) < 299800):
                    print("Passing to the model")
                    output = cv2.resize(input_img, (28, 28))
                    plt.imshow(output)
                    plt.show()

                input_img = np.full((480, 640), 255, dtype=np.uint8)

            for hand in hands:
                drawing_utils.draw_landmarks(rgb_frame, hand) 
                landmarks = hand.landmark

                for id, landmark in enumerate(landmarks):
                    x = int((landmark.x) * frame_width)
                    y = int((landmark.y) * frame_height)

                    if id == 8:
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        cv2.circle(input_img, (x,y), radius=20, color=(0, 0, 0), thickness=-1)

                        points.append([x, y])
                        if (len(points) == 2):
                            # cv2.line(input_img, points[0], points[1], color=(0, 0, 0), thickness=30)
                            points.pop(0)
                            # points=[]
                        print((x, y), end="")
                    if id==12:
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        print((x, y))


            prev=cur

        cv2.imshow("img", input_img)
        cv2.imshow("PBL Gesture", rgb_frame) 
        
        if cv2.waitKey(1) & 0xFF==27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
