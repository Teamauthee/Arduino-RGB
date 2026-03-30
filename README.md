# Arduino-RGB

An end-to-end machine learning pipeline that classifies RGB colors and runs inference
on embedded hardware.

## Overview

This project explores a full ML pipeline from unsupervised labeling to embedded
deployment using RGB color data as the domain.

### Pipeline

**1. Labeling with K-Means**
Raw RGB values are clustered into three categories (red, green, blue) using K-Means.
No manual annotation needed — the algorithm discovers the color groupings automatically.

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

## Structure
├── data/          # Raw RGB color samples
├── clustering/    # K-Means labeling script
├── model/         # PyTorch training + weight export
├── arduino/       # C++ inference sketch
└── README.md