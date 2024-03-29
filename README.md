# BrainPad

![](pictures/logo_large.png "Logo")

Classification of EEG signals from the brain through OpenBCI hardware and Tensorflow-Keras API.

<p align='center'>
  <img src="pictures/helmet_orig.jpg" />
</p>
<p align='center'>
  <img src="pictures/short_demo.gif" width="400" /> 
</p>

## Table of Contents

* [Disclaimer](#disclaimer)
* [Data Acquisition](#data-acquisition)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Usage](#usage)
* [Confusion Matrix](#confusion-matrix-so-far)
* [A look at the samples](#a-look-at-our-samples)
* [Best NN so far](#the-best-neural-network-so-far)

## Disclaimer
This is a boilerplate work-in-progress project for motor imagery classification with deep learning using OpenBCI Cyton board.  
Feel free to take inspiration and use the code.  
Don't forget to cite me and the articles that have had a huge impact on this project if you will use them.
Please let me know if you find any improvements.

## Data Acquisition
The ```personal_dataset``` folder provides the current EEG samples taken following this protocol:<br>
* The person sits in a comfortable position on a chair and follows the `acquire_eeg.py`
protocol. 
* When the program tells to think "hands" the subject imagines opening and closing
both hands. 
* If "none" is presented the subject can wonder, and think at something else. 
* If "feet" is presented the subject imagines moving the feet up and down. <br>

The subject does not blink during acquisitions.

Each sample is stored as a numpy 2D array in an .npy file that has the following shape:<br>
`(8, 250)`

## Prerequisites

To get a local copy up and running follow these simple steps.

The project provides a ```Pipfile``` file that can be managed with [pipenv](https://github.com/pypa/pipenv).  
```pipenv``` installation is **strongly encouraged** in order to avoid dependency/reproducibility problems.

* pipenv

```sh
pip install pipenv
```

## Installation

1. Clone the repo

```sh
git clone https://github.com/CrisSherban/BrainPad
```

2. Enter in the project directory and install Python dependencies

```sh
cd BrainPad
pipenv install
```




## Usage
* #### acquire_eeg.py
    This script allows to connect to OpenBCI Cyton board through BrainFlow
    and acquire data in form of raw EEG. <br>
    For a Cyton board and a Linux machine the setup is the following:
    *   Connect the Ultracortex Helmet with the Cyton Board to your machine
    *   Run the script and follow the acquisition protocol
    
* #### live_test_brainflow.py
    This Python module gives the user a live testing environment of the system. <br>
    For a Cyton board and a Linux machine the setup is the following:
    *   Connect the Ultracortex Helmet with the Cyton Board to your machine
    *   Open OpenBCI GUI
    *   Set this script in the OpenBCI GUI Working Directory
    *   Mimic the motor imagery tasks you did in the acquisition protocol and check on screen what happens.
    
* #### dataset_tools.py
    This module provides functionalities for splitting a dataset, loading a dataset
    visualizing data, and handles all the necessary preprocessing.
    
* #### check_model.py
    Allows to check how well a model is doing on some unseen set of data.
    
* #### neural_nets.py
    Provides three different architectures used in this project. 
    * A very deep architecture: ResNet
    * A simplistic architecture based upon the knowledge from: <br> https://iopscience.iop.org/article/10.1088/1741-2552/ab0ab5/meta 
    * TA-CSPNN made for motor imagery classification tasks, all credits to:
     <br> https://github.com/mahtamsv/TA-CSPNN/blob/master/TA_CSPNN.py
     <br> https://ieeexplore.ieee.org/document/8857423
    * EEGNet: A Compact Convolutional Network for EEG-based Brain-Computer Interfaces:
     <br> https://arxiv.org/abs/1611.08024
  
* #### training.py
    Provides several functions to train the networks in Keras. 

* #### model_optimization.py
    Gives a sketch on how to use Keras tuner and GridSearch to tune the hyperparameters.

    
    
## Confusion Matrix so far:
<p align='center'>
<img src="pictures/confusion_matrix.png">

## A look at our samples:
<p align='center'>
<img src="pictures/before.png">
<img src="pictures/after_std.png">
<img src="pictures/after_bandpass.png">
<img src="pictures/ffts.png">
</p>

## The Best Neural Network so far:
<p align='center'>
<img src="pictures/net.png">
</p>
