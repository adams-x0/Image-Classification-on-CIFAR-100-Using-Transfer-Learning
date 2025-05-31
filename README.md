# IMAGE CLASSIFICATION ON CIFAR-100 USING TRANSFER LEARNING WITH EFFICIENTNET-B0

## INTRODUCTION / BACKGROUND

CIFAR-100 is a labeled subset of 80 million tiny images dataset where CIFAR stands for Canadian Institute For Advanced Research. The dataset consists of 60000 colored images (50000 training and 10000 test) of  pixels in 100 classes grouped into 20 super classes. Each image has a fine label (class) and a coarse label (superclass).

## DATASET

The python version of this dataset can be downloaded from the website of the University of Toronto. The downloaded files are python pickled objects. It has 100 classes containing 600 images each. There are 500 training images and 100 testing images per class.

## METHODS

I employed transfer learning using EfficientNet-B0 architecture pre-trained on ImageNet. The CIFAR-100 images were resized from  to match EfficientNet’s input requirements. A custom data generator was implemented to handle image resizing, normalization, and optional augmentation. The model was fine-tuned by adding a global average pooling layer, a dropout layer to prevent overfitting, and a final den se layer with softmax activation for 100-class classification. The network was trained using the Adam optimizer, with categorical crossentropy loss, and performance was monitored using accuracy on both training and validation sets. Early stopping and learning rate reduction on plateau were used to optimize training.

## EXPERIMENTS

I began by building and training custom Convolutional Neural Network (CNN) models from scratch on the CIFAR-100 dataset, aiming to achieve reasonable classification accuracy. I experimented with different configurations such as varying number of convolutional layers, filter sizes, dropout rates, and learning rates. Despite these efforts, the models performed poorly, and accuracy remained low on both the training and validation sets (36%).

To better understand how to approach this task, I researched existing solutions online and came across a blog post by Chetna Khanna titled CIFAR 100: Transfer Learning using EfficientNet on Towards Data Science. I closely followed this approach. After implementing the same architecture and training strategy, I was able to replicate the reported performance and achieve higher accuracy (Test Accuracy:  81.18 %) compared to my earlier attempts.

## RESULTS

The model was evaluated using accuracy as the primary metric on the validation and test sets. After training EfficientNet-B0 for 15 epochs using transfer learning, the best validation accuracy achieved was approximately 82%, while the final test accuracy was 81%.

The training process was monitored using loss and accuracy plots. As shown in Figure 1, training and validation loss consistently decreased while accuracy steadily increased, indicating effective learning without severe overfitting. A confusion matrix was generated to analyze per-class performance, revealing that certain classes had significantly higher prediction accuracy than others – likely due to visual similarity.

Additionally, a set of sample predictions was visualized to compare predicted vs. true labels, giving qualitative insight into the model’s strengths and weaknesses across different categories.

![Figure 1](#)
![Figure 2](#)

## DISCUSSION AND NEW IDEAS

Deep learning relies heavily on experimentation. While EfficientNet-B0 performed well, accuracy could potentially be improved by using larger variants like EfficientNet-B3 or B5, or through more extensive hyperparameter tuning. Techniques such as adjusting learning rates or using advanced augmentation could also enhance performance.

Another promising direction is applying Vision Transformers (ViTs) which have recently outperformed CNNs in many image classification tasks. Exploring transformer-based models on CIFAR-100 could offer improved results and a modern alternative to convolutional architectures.

## References

Khanna, C. (2021, January 1). CIFAR-100: Pre-processing for image recognition task. Towards Data Science. https://towardsdatascience.com/cifar-100-pre-processing-for-image-recognition-task-68015b43d658/

Khanna, C. (2021, March 30). CIFAR-100 Transfer Learning using EfficientNet. Towards Data Science. https://towardsdatascience.com/cifar-100-transfer-learning-using-efficientnet-ed3ed7b89af2/

Krizhevsky, A. (2009). Learning multiple layers of features from tiny images. University of Toronto. https://www.cs.toronto.edu/~kriz/cifar.html
