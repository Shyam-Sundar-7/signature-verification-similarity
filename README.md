# Handwritten Signature Authenticity Verification System

## Description:
This project aims to develop a machine learning model for automatically verifying the authenticity of handwritten signatures, thereby streamlining the signature validation process and reducing errors.

## Problem Statement:
Manual signature verification is a time-consuming and error-prone process. The goal of this project is to create a machine learning model capable of distinguishing between genuine and forged handwritten signatures, enabling faster and more accurate signature validation processes.

## Problem Description:
The task is to train a machine learning model to differentiate between genuine and forged handwritten signatures based on a dataset of signature images. The model should return a match percentage between the two input signatures.

## Implementation:
- **SiasemeNetwork1 (PyTorch Model)**
  - Binary Cross Entropy as the loss function
  - Two input images are passed through a shared weights CNN model
  - The final vector is the square distance of the CNN output, which is then passed to a feed-forward network with a sigmoid to the last layer for classification
  - Codes for the model in the code folder:
    - Model1.py - The codes for the pytorch and pytorch lightning structures
    - training1.py - training code
    - inference1.py - Indference code

- **SiameseNetwork2 (PyTorch Model)**
  - Contrastive Loss as the loss function
  - Two images are sent to the same CNN and followed by a feed-forward network with shared weights
  - The last layer of the feed-forward network acts as a feature extractor to distinguish similar images
  - Codes for the model in the code folder:
    - Model2.py - The codes for the pytorch and pytorch lightning structures
    - training2.py - training code
    - inference2.py - Indference code



## Installation:

1. Clone the repository to the local machine: 
    ``` 
    git clone https://github.com/Shyam-Sundar-7/signature-verification-similarity.git
    ```

2. Create a virtual environment and activate it and install requirements.txt:
    ```
    python -m venv env
    source env/bin/activate
    pip install -r requirements.txt
    ```


3. Execute the `data_generation_to_csv.ipynb` notebook for a structured way of splitting the dataset into training, validation, and testing sets.

4. Execute the `execution.ipynb` notebook for the execution of the training of the models and prediction of the models.

5. To run the streamlit application.
    ```
    streamlit run main.py
    ```

# Short Demo of the streamlit application


https://github.com/Shyam-Sundar-7/signature-verification-similarity/assets/101181076/7119939f-3512-4d8b-a9fb-d91f11b80a3f

