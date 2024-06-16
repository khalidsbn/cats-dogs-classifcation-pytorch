<div align="center">
    <h1>Animals classification:</h1>
</div>


## Introduction:
This project aims to develop an algorithm to classify images as containing either a dog or a cat. While this task is straightforward for humans and our pets, it poses a significant challenge for computers. Leveraging the dataset from the competition, we aim to build and train a binary classification model using PyTorch. This process involves preprocessing the image data, designing a neural network architecture, and fine-tuning the model to distinguish between images of dogs and cats accurately. Through this project, we explore various deep learning techniques and demonstrate the application of PyTorch in solving a real-world image classification problem.

## A directory structure: 
```
going_modular/
├── binary_classification_with_PyTorch.ipynb
├── script_mode.ipynb
├── modular/
│   ├── data_setup.py
│   ├── engine.py
│   ├── model_builder.py
│   ├── train.py
│   └── utils.py
├── models/
│   └── CNN_model.pth
└── data/
    └── train/
    │   ├── cat.225.jpg
    │   ├── dog.123.jpg
    │   └── ...
    ├── valid/
    │   ├── cat.225.jpg
    │   ├── dog.123.jpg
    │   └── ...
    └── test/
        ├── 8.jpg
        ├── 9.jpg
        └── ...
        
```
**modular**: This folder contains all the essential code for the project.
**binary_classification_with_PyTorch.ipynb**: The main Jupyter notebook that retrieves the dataset using the Kaggle API.
**script_mode.ipynb**: A streamlined Jupyter notebook containing only the necessary code to run and train the model.

## CNN Architecture:
The CNN architecture used to train the model can be broken down into:
* Input Layer
* Layer 1: Convolutional + BatchNorm + ReLU + MaxPooling
* Layer 2: Convolutional + BatchNorm + ReLU + MaxPooling
* Layer 3: Convolutional + BatchNorm + ReLU + MaxPooling
* Flattening Layer
* Fully Connected Layer 1
* Fully Connected Layer 2

## How to Run the model:
1. Colone this repo
```
git clone https://github.com/khalidsbn/cats-dogs-classifcation-pytorch.git
```
2. Create and activate a virtual environment
```
python3 -m venv .venv
source .venv/bin/activate
```
3. Install the required dependencies (pandas, Numpy, etc)
```
pip install -r requirements.txt
```
4. Train the model
Open project on terminal, and run:
```
!python modular/train.py
```