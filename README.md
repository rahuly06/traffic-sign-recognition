# Traffic Sign Recognition using PyTorch
A computer vision project for recognizing German traffic signs (GTSRB dataset) using a Convolutional Neural Network (CNN). This project demonstrates end-to-end training, evaluation, and visualization using PyTorch.

## Dataset
GTSRB (German Traffic Sign Recognition Benchmark). Training images are organized into subfolders for each class (0,1,...,42). Test images are provided in a flat folder, with labels in Test.csv.

## Key Concepts
ImageFolder: Automatically assigns labels based on subfolder names for training. Custom Dataset: Needed for test images stored in a flat folder with labels in CSV. CNN Architecture: Two convolutional layers, max pooling, followed by fully connected layers and dropout.

## Results
Achieved >99% test accuracy on the GTSRB dataset. High accuracy is expected due to clean, well-cropped traffic sign images.

## References
GTSRB dataset on Kaggle  <br />
PyTorch ImageFolder Documentation
