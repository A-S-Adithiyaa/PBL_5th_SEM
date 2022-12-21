import cv2
import mediapipe as mp
import matplotlib.pyplot as plt
from tensorflow import keras
import numpy as np
import pickle

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

        # img_no=1

        if hands:
            cur=len(hands)
            # if (cur == 2 and prev == 2):
            if (middle_y<index_y):
                if (np.count_nonzero(input_img) > 3400):
                    
                    
                    input_img_resized = cv2.resize(input_img, (28, 28))
                    input_img_reshaped=input_img_resized.reshape((1, 28, 28, 1))
                    # print(input_img_reshaped.shape, type(input_img_reshaped))
                    value = pickled_model.predict(input_img_reshaped)
                    print(value.argmax())
                    plt.imshow(input_img_resized)
                    plt.show()
                    # filename="C"+str(img_no)+".jpg"
                    # cv2.imwrite(filename, input_img_resized)
                    # img_no+=1
                    # # print(np.count_nonzero(input_img))
                    # print("Passing to the model")
                    # print(input_img, len(input_img), type(input_img))
                    # input_img_resized = cv2.resize(input_img, (28, 28))
                    # input_img_resized=input_img.reshape((28, 28))
                    # input_img=np.transpose(input_img)
                    # input_img_resized = cv2.resize(input_img, (28, 28))
                    # input_img_resized_final=input_img_resized.reshape(1, 28, 28, 1)
                    # print(input_img_resized_final.shape)
                    # value = pickled_model.predict(input_img_resized_final)
                    # print(value, type(value))
                    # # value_list=list(value)
                    # # print(value_list[0].index(1))
                    # # output = cv2.resize(input_img, (28, 28))
                    # # pickled_model = pickle.load(open('pblModel.pkl', 'rb'))
                    # pickled_model.predict(output)
                    # plt.imshow(input_img_resized)
                    # plt.show()
                    points.clear()


                input_img = np.full((480, 640), 0, dtype=np.uint8)

            for hand in hands:
                drawing_utils.draw_landmarks(rgb_frame, hand) 
                landmarks = hand.landmark

                for id, landmark in enumerate(landmarks):
                    x = int((landmark.x) * frame_width)
                    y = int((landmark.y) * frame_height)

                    if id == 8:
                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        # cv2.circle(input_img, (x,y), radius=20, color=(0, 0, 0), thickness=-1)

                        index_y=y

                        points.append([x, y])
                        if (len(points) == 2):
                            cv2.line(input_img, points[0], points[1], color=(255, 255, 255), thickness=30)
                            points.pop(0)
                            # points=[]
                        # print((x, y), end="")
                    if id==12:
                        middle_y=y

                        cv2.circle(img=rgb_frame, center=(x, y), radius=10, color=(0, 255, 255))
                        # print((x, y))
            prev=cur

        cv2.imshow("img", input_img)
        cv2.imshow("PBL Gesture", rgb_frame) 
        
        if cv2.waitKey(1) & 0xFF==27:
            break


    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    # pickled_model = pickle.load(open('planB/pblModel.pkl', 'rb'))
    pickled_model = keras.models.load_model('planB/my_model.h5')
    main()

# import pickle

# pickled_model = pickle.load(open('planB/pblModel.pkl', 'rb'))
# pickled_model.summary()
