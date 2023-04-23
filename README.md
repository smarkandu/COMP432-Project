# COMP432-Project

## COMP 432: Circuit Component Image Identifier
## By Steven Markandu

### **Abstract**
The goal of this project was to train a model that takes hand drawn circuit component images as input and is able to properly classify them using the label given to us via the data set.  The data set was obtained via Kaggle, and contains the necessary images which can be used for both training the model and then finally testing it.  CNNs was used in implementing this.  

The main challenges include having enough data to sufficiently test/train my model.  A huge challenge will be in researching a sufficient machine learning model(s) different from CNN in order to accomplish the same task.  And finally, fine-tuning our model in order to achieve the best performance we’re able to achieve.  

The performance metrics include:

*   Accuracy
*   Error
*   Loss per epoch (using CrossEntropyLoss)

We created three models for this problem.  They were:

1. Plain CNN
2. LeNet-5
3. AlexNet

The Plain CNN performed decently with a test accuracy of 86.907%, but the graph was slighly erratic.  The LeNet-5 model had a slightly lower accuracy of 77.2%, but was way smoother with less oscillations.  And finally, the AlexNet model proved to be a disappointment with only 65.9% test accuracy and wild oscillations for the test curve.

Based on our results, we concluded that the LeNet-5 was the better model in this project due to its decent accuracy score, it's more stable curve, and it being an established model.

