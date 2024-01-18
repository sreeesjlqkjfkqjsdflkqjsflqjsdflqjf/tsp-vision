import cv2
import mediapipe as mp

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()

pad = 20

#Permet de crop IR-Camera
def crop_hand(input_img):
    frame = cv2.imread(input_img)
    grayImage = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret,thresh1 = cv2.threshold(grayImage,127,255,cv2.THRESH_BINARY)
    min_x, max_x, max_y = 9999, -9999, -9999

    for x in range(thresh1.shape[1]):
        for y in range(thresh1.shape[0]):
            current_value = thresh1[y][x]
            if current_value != 0:
                min_x = min(x, min_x)
                max_x = max(x, max_x)
                max_y = max(y, max_y)
    
    output = frame[0:max_y+pad,min_x-pad:max_x+pad]

    return output #Output : image np-array

#Permet de crop RGB-Camera
def crop_hand_mp(input_img): #Input : path vers photo (string)
    frame = cv2.imread(input_img)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)
    if results.multi_hand_landmarks:
        hand_landmarks = results.multi_hand_landmarks[0].landmark

        x_min, y_min = float('inf'), float('inf')
        x_max, y_max = float('-inf'), float('-inf')

        for landmark in hand_landmarks:
            x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
            x_min = min(x_min, x) 
            y_min = min(y_min, y) 
            x_max = max(x_max, x) 
            y_max = max(y_max, y)

        if x_max > x_min and y_max > y_min:
            hand_cropped = frame[y_min-pad:y_max+pad, x_min-pad:x_max+pad]
            return hand_cropped
        else:
            return frame #Output : image np-array
