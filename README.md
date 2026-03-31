# Arduino-RGB

An end-to-end machine learning pipeline that classifies RGB colors and runs inference
on embedded hardware.

## Overview

This project explores a full ML pipeline from unsupervised labeling to embedded
deployment using RGB color data as the domain.

### Pipeline

**1. Labeling with K-Means**
Raw RGB values are clustered into three categories (red, green, blue) using K-Means.
No manual annotation needed, the algorithm discovers the color groupings automatically.

**2. Training with PyTorch**
A lightweight neural network is trained on the labeled data to learn the decision
boundaries between the three color classes. The trained weights are then exported.

**3. Inference on Arduino (C++)**
The exported weights are hardcoded into a C++ sketch. Given an RGB triplet, the model
runs a forward pass on the microcontroller and predicts the color category. The
corresponding LED (red, green, or blue) on the breadboard lights up to confirm the
prediction.

## Stack
- Python · scikit-learn · PyTorch
- Arduino (C++) · Breadboard · RGB LEDs

## Conclusion
### Learning
For this first Arduino project that is not in the starterkit book, I learnt how to labeled data with Unsupervised Learning Algorithm (Clustering), how to feed this data into a Neural Network (with 275 trainable parameters) with an accuracy of 87.65% (Which is bad for classify colors) and then export the model's weights into the Arduino Uno in order to predict to which class color a random RGB code (color) belongs to. 

### Problems
The accuracy score is pretty low for just predicting to which class color they belong, I could first add some data (other colors with their RGB code) but mostly I could use another clustering algorithm like DBSCAN or MeanShift but as we can expect with Kmeans is that the random first centroid plays a huge role when clustering the data in three classes (My best result was 92.17% accuracy). I could play with more variety of NN architechture (I tested three of them including the one in nn.py) but I needed a light NN architechture where weights can fit into the SRAM of the Arduino Uno (2 Ko), so I was limited in choice that is why on the testing colors in the c++ code, I get 50% accuracy with RGB LEDs (see Schema). At the end, I was satisfied with the results as the goal was to get things working and it does.
