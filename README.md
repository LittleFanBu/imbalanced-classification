# imbalanced-classification
Imbalanced Classification: one of the class is relatively rare as compared with other class(es)
One of the ways to alleviate imbalanced problem is to use data-augmentation, such as flip, rotate, crop, 
and add noise) to generate the samples of the minority class of the training set, helping the model to learn 
rich representations of minority class
In this task,I implemented a deep learning model AlexNet to solve an imbalanced face mask detection
I use the following methods to do data-augmentation, each methods will generate 60 new without mask data, and generate 420 new data totally:
transforms.RandomHorizontalFlip(p=1)
transforms.RandomVerticalFlip(p=1)
transforms.RandomRotation([60, 90])
transforms.RandomRotation([100, 180])
transforms.RandomRotation([200, 270])
transforms.GaussianBlur(3)
transforms.RandomInvert(p=1)
Then, I use “ConcatDataset” to link the new data with the original minority data and with mask data together as the training data
And the result:
<img width="415" alt="image" src="https://user-images.githubusercontent.com/121480302/209661010-30029385-38fb-4cf6-b2a1-1a49f8952fae.png">

convergence curve of Model without data-augmentation

<img width="415" alt="image" src="https://user-images.githubusercontent.com/121480302/209660899-b0f43a36-d88d-4561-8b8a-d0a4073462a1.png">
convergence curve of Model with data-augmentation(500:480)

In conclusion, after doing data-augmentation for minority class, the model can get better G_mean and its curve converges quickly.

