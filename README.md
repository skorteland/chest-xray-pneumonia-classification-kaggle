# Chest X-ray Pneumonia Classifier

This project explores several convolutional neural networks to detect pneumonia from chest X-ray images using Pytorch.

## Dataset
- [Kaggle Chest X-ray Pneumonia Dataset] (https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- 5863 X-ray images in two categories (Pneumonia and Normal)

## Models
In this project relatively simple models were used as only a CPU was available for training.

1. tiny CNN model consisting of two convolutional layers and one fully coupled layer.
2. simple CNN model consisting of three convolutional layers and two fully coupled layers.
3. transfer learning with a pre-trained ResNet18 and Resnet34 model
4. transfer learning with a pre-trained MobileNetV2 model
5. transfer learning with a pre-trained EfficientNet-b0 model

The models are described and explored in the notebooks.