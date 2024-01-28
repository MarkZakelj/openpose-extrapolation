# OpenPose Skeleton Keypoiny Extrapolation

## Overview
This repository hosts a streamlined neural network designed specifically for extrapolating non-visible keypoints in OpenPose skeleton representations. It serves as a robust solution for enhancing keypoint detection, particularly in scenarios where certain body parts are obscured or off-screen.

## Contents
- `preprocess.ipynb`: A detailed Jupyter notebook illustrating the data preprocessing steps. It provides a comprehensive guide on how the data was prepared for training the neural network.
- `train.py`: The core script for training the neural network. This file encapsulates the training process, including parameter settings, model architecture, and optimization routines.
- `predict.py`: A FastAPI application enabling easy and efficient model inference. This script turns the neural network into a web-service, allowing for real-time keypoint extrapolation through a simple API.

## Getting Started
To begin using this repository, clone it to your local machine and follow the setup instructions detailed in each script. The `preprocess.ipynb` notebook offers an excellent starting point to understand the data preparation, while `train.py` and `predict.py` guide you through training the model and deploying it for inference, respectively.


## Reason for development
Utilizing the [dw-pose](https://github.com/IDEA-Research/DWPose) model for OpenPose keypoint detection, we occasionally encounter limitations in keypoints prediction, especially in scenarios where the subject is only partially visible (e.g., from the torso upwards) or when parts of the body, like a hand, are outside the frame. This poses a challenge in applications like generating OpenPose conditioning images for ControlNet in stable diffusion contexts, where a full-body skeleton representation is preferable.

To address these constraints, we introduce an enhancement model designed to operate as a post-processing step following dw-pose predictions. This model is particularly adept at inferring and reconstructing the full skeleton, even in cases where parts of the subject are off-screen or obscured. By employing this model, we can extend the capabilities of dw-pose, enabling more comprehensive and accurate skeleton predictions suitable for a wider range of applications, including those requiring full-body representations.


