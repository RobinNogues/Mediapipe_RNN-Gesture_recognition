import cv2
import mediapipe as mp
import keyboard
import numpy as np
from time import time
from keras.models import load_model
from config import write_on_image, searching_for_hands, get_coordinates_hands, draw_hands, add_gesture_dataset, read_dataset


def start_detection(n_frames_to_detect=5):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0)
    cap = cv2.VideoCapture(0)
    X, actions = read_dataset()[::2]
    model = load_model("gesture.h5")
    X_test = np.zeros((X[0].shape[0], 126))

    n = 1
    pred = ""
    while cap.isOpened():
        image = cap.read()[1]
        image = cv2.flip(image, 1)
        results = searching_for_hands(hands, image)
        if results.multi_hand_landmarks:
            # image = draw_hands(results, image, mp_hands)
            coord = get_coordinates_hands(results)
            X_test = np.roll(X_test, 1, 0)
            X_test[0] = coord

            if n == n_frames_to_detect:
                n = 1
                res = model.predict(np.array([X_test]))
                pred = actions[np.argmax(res[0])]
                image = write_on_image(image, pred)
            elif pred:
                n += 1
                image = write_on_image(image, pred)
            else:
                n += 1
        cv2.imshow('Camera', image)
        if cv2.waitKey(1) & keyboard.is_pressed('esc'):
            break
    cap.release()
    cv2.destroyAllWindows()


def add_to_dataset(label, i_instance=100, k_frames=20, duration_pause=0.0):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=2, model_complexity=0)
    cap = cv2.VideoCapture(0)
    started = ended = start_countdown = start_pause = k = i = 0
    all_instances = np.zeros([i_instance, k_frames, 126])
    info = "press s to start"
    while cap.isOpened():
        t = time()
        image = cap.read()[1]
        image = cv2.flip(image, 1)
        results = searching_for_hands(hands, image)
        image = draw_hands(results, image, mp_hands)
        time_is_up = (t - start_countdown >= 3) if started else False
        if started and time_is_up and results.multi_hand_landmarks and i < i_instance:
            if t - start_pause < duration_pause:
                info = "Pause : {}sec left".format(round(duration_pause-(t-start_pause), 1))
            elif k < k_frames:
                info = "Instance: {}".format(i + 1)
                coord = get_coordinates_hands(results)
                all_instances[i, k] = coord
                k += 1
            else:
                k = 0
                i += 1
                start_pause = t
        elif not ended and i == i_instance:
            ended = True
            info = add_gesture_dataset(all_instances, label)
        elif started and not time_is_up:
            info = "Start in : {}s".format(int(3.99 - (t - start_countdown)))
        elif keyboard.is_pressed("s") and not started:
            start_countdown = time()
            started = True

        if cv2.waitKey(1) & keyboard.is_pressed('esc'):
            break
        image = write_on_image(image, info)
        cv2.imshow('Caméra', image)
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    start_detection(5)
    # add_to_dataset("4", 100, 20, 1)

