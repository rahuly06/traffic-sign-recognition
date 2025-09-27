# Traffic Sign Recognition using PyTorch
A computer vision project for recognizing German traffic signs (GTSRB dataset) using a Convolutional Neural Network (CNN). This project demonstrates end-to-end training, evaluation, and visualization using PyTorch.

## Dataset
GTSRB (German Traffic Sign Recognition Benchmark). Training images are organized into subfolders for each class (0,1,...,42). Test images are provided in a flat folder, with labels in Test.csv.

## Key Concepts
- **ImageFolder**: Automatically assigns labels based on subfolder names for training.  
- **Custom Dataset**: Needed for test images stored in a flat folder with labels in CSV.  
- **CNN Architecture**: Two convolutional layers, max pooling, followed by fully connected layers and dropout.

## Results
Achieved >99% test accuracy on the GTSRB dataset. High accuracy is expected due to clean, well-cropped traffic sign images.

## Combining YOLO and CNN
YOLOv8 was used for object detection in images or video. The CNN model from the first part of the project was used to classify the detected traffic signs.

## Output Video
You can view the processed traffic sign recognition video below:

<video width="640" height="480" controls>
  <source src="video/output_with_cnn.mp4" type="video/mp4">
  Your browser does not support the video tag.
</video>

Alternatively, download and view: [output_with_cnn.mp4](video/output_with_cnn.mp4)

## References
- [GTSRB dataset on Kaggle](https://www.kaggle.com/datasets/meowmeowmeowmeowmeow/gtsrb-german-traffic-sign)  
- [PyTorch ImageFolder Documentation](https://pytorch.org/vision/stable/generated/torchvision.datasets.ImageFolder.html)
