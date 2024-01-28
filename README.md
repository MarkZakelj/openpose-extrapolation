# OpenPose Skeleton Keypoiny Extrapolation

## Overview
This repository hosts a streamlined neural network designed specifically for extrapolating non-visible keypoints in OpenPose skeleton representations. It serves as a robust solution for enhancing keypoint detection, particularly in scenarios where certain body parts are obscured or off-screen.

## Getting Started
- Create a virtual environment and install the requirements using `pip install -r requirements.txt`.
- Run `preprocess.ipynb` notebook to generate augumented data.  
- Run `train.py` to train the model. Be sure, to match the data version Change the file to change the hyperparameters and data version (`DATA_VERSION` variable).
- Run `tensorboard --logdir lightning_logs` to view the training progress.  
- Copy the preffered model from lightning logs to `models` dir and change the model name in  `predict.py`.  
- Use `uvicorn` to run a prediction API - `uvicorn predict:app --port 3000`

## Data source
The data used `data/poses.csv` was found on [kaggle](https://www.kaggle.com/datasets/pashupatigupta/human-keypoints-tracking-dataset). It contains 2D keypoints of 18 body parts and a class representing the human activity, which was not used in our model.

## Reason for development
Utilizing the [dw-pose](https://github.com/IDEA-Research/DWPose) model for OpenPose keypoint detection, we occasionally encounter limitations in keypoints prediction, especially in scenarios where the subject is only partially visible (e.g., from the torso upwards) or when parts of the body, like a hand, are outside the frame. This poses a challenge in applications like generating OpenPose conditioning images for ControlNet in stable diffusion contexts, where a full-body skeleton representation is preferable.

To address these constraints, we introduce an enhancement model designed to operate as a post-processing step following dw-pose predictions. This model is particularly adept at inferring and reconstructing the full skeleton, even in cases where parts of the subject are off-screen or obscured. By employing this model, we can extend the capabilities of dw-pose, enabling more comprehensive and accurate skeleton predictions suitable for a wider range of applications, including those requiring full-body representations.


