# Mediapipe / RNN - Gestures recognition
This program can recognize gestures of the dataset from both hands. Hands or located with the library Mediapipe, and vectorised into 21 coordinates for each hand. These coordinates are recognised as gestures with the recurrent neural network.

Here are 2 samples videos for:
- gestures recognition : https://youtu.be/WMdoohGD50E
- gesture added to the dataset : https://youtu.be/f5fNRzVzyak

## Libraries
- tensorflow
- mediapipe
- opencv
- numpy
- keyboard

## Tutorial
- To predict gestures :
main.py, start start_detection() with one parameter which specifies the frequency of detection (n frames to predict) (int, default 5)

- To add gestures :
    - main.py, start add_to_dataset() with 4 parameters :
	    - name of gesture (string)
	    - number of instances (int, default 100) 
	    - number of frames per gesture (int, default 20)
	    - pause between 2 instances (double, default 0)
    - model.py, start train() to train the model

## Students
This program has been made by Robin NOGUES, Hugo BOURREAU, Cyril GUIRGUIS and Alex DEMARS as part of the studies at UQAC.




