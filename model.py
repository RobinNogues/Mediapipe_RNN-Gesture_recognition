import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.callbacks import TensorBoard
import os
from config import read_dataset


def train():
    X, y, labels = read_dataset()
    y = to_categorical(y).astype(int)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    log_dir = os.path.join('Logs')
    tb_callback = TensorBoard(log_dir=log_dir)

    model = Sequential()
    model.add(Input(shape=X_train[0].shape))
    model.add(LSTM(32, return_sequences=True, activation='relu'))
    model.add(LSTM(16, return_sequences=False, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(labels.shape[0], activation='softmax'))

    model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    model.fit(X_train, y_train, epochs=30, callbacks=[tb_callback])

    model.save('gesture.h5')


if __name__ == "__main__":
    train()
