# BrainPad
Classification of EEG signals form the brain through OpenBCI hardware and Tensorflow-Keras API
<img src="pictures/helmet.jpg">

## Usage
*   #### acquire_fft.py
    This script allows to connect to OpenBCI GUI
    through LSL Protocol and acquire data in form of FFT. <br>
    For a Cyton board and a Linux machine the setup is the following:
    *   Connect the Ultracortex Helmet with the Cyton Board to your machine
    *   Open OpenBCI GUI
    *   Set this script in the OpenBCI GUI Working Directory
    *   Set in the script the type of acquisition you want, for example [Left, Right, None]
    *   Think at the chosen action and press Enter
 
*   #### live_test.py
    This Python module gives the user a live testing environment of the system. <br>
    For a Cyton board and a Linux machine the setup is the following:
    *   Connect the Ultracortex Helmet with the Cyton Board to your machine
    *   Open OpenBCI GUI
    *   Set this script in the OpenBCI GUI Working Directory
    *   Think at some action from [Left, Right, None] and check on screen what happens
    
## Confusion Matrix so far:
<img src="pictures/confusion_matrix.png">

## The Neural Network so far:
<img src="pictures/crisnet.png">