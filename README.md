# Mediapipe / RNN - Gesture recognition
This program can recognize gestures of the dataset from both hands. Hands or located with the library Mediapipe, and vectorised into 21 coordinates for each hand. These coordinates are recognised as gestures with the recurrent neural network.

Here are 2 samples videos for:
- gesture recognition : https://youtu.be/WMdoohGD50E
- gesture added to the dataset : https://youtu.be/f5fNRzVzyak

## Libraries
- tensorflow
- mediapipe
- opencv
- numpy
- keyboard

## Tutorial
- To predict gestures :
main.py, use start_detection() with one parameter which specifies the frequency of detection (n frames to predict) (int, default 5)

- To add gestures :
    - main.py, use add_to_dataset() with 4 parameters :
	    - name of gesture (string)
	    - number of instances (int, default 100) 
	    - number of frames per gesture (int, default 20)
	    - pause between 2 instances (double, default 0)
    - model.py, use train() to train the model

## Informations
The model and the dataset in this repository are those used in the demonstration video.

## Students
This program made by Robin NOGUES, is part of a larger project to discover severals models for gestures recognition (CNN, Transfer Learning, Random Forest and RNN), by Hugo BOURREAU, Alex DEMARS, Cyril GUIRGUIS and Robin NOGUES as part of the studies at UQAC.






