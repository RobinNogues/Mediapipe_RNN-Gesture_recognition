import cv2
import mediapipe as mp
import numpy as np
from os.path import exists


def searching_for_hands(hands, image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    return results


def get_coordinates_hands(results):
    coord = np.zeros([2, 63])
    list_hands = results.multi_hand_landmarks
    list_description_hands = results.multi_handedness
    for n, hand in enumerate(list_hands):
        hand_description = list_description_hands[n].classification[0]
        coord[hand_description.index] = np.array([[lm.x, lm.y, lm.z] for lm in hand.landmark]).flatten()
    return coord.flatten()


def flip_hand_coordinate(flat_hand_instance):
    for index, frame in enumerate(flat_hand_instance):
        new_coord = np.array([[flat_hand_instance[i],
                              flat_hand_instance[i+1],
                              flat_hand_instance[i+2]]
                             for i in range(len(flat_hand_instance-2))])
        flat_hand_instance[index] = new_coord
    return flat_hand_instance


def draw_hands(results, image, mp_hands):
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
    return image


def write_on_image(image, gesture):
    gesture = str(gesture)
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.7
    org = (50, 50)
    color = (255, 0, 0)
    thickness = 2
    return cv2.putText(image, gesture, org, font,
                       font_scale, color, thickness, cv2.LINE_AA)


def read_dataset():
    if exists("dataset.npz"):
        data = np.load('dataset.npz')
        return data['X'], data['y'], data['labels']
    else:
        return [], [], []


def add_gesture_dataset(X_instances, class_label):
    if exists("dataset.npz"):
        X, y, labels = read_dataset()
        if X[0].shape == X_instances[0].shape:
            np.savez('dataset_backup.npz', X=X, y=y, labels=labels)
            y_instances = np.full(X_instances.shape[0], labels.size)
            X = np.concatenate((X, X_instances))
            y = np.concatenate((y, y_instances))
            labels = np.append(labels, class_label)
            np.savez('dataset.npz', X=X, y=y, labels=labels)
            return "Added to dataset"
        else:
            return "Error n frames: {} expected".format(X[0].shape[0])
    else:
        y = np.zeros(X_instances.shape[0])
        np.savez('dataset.npz', X=X_instances, y=y, labels=np.array([class_label]))
        return "Dataset created"


if __name__ == "__main__":
    print(read_dataset()[0][0][0])
