# ACGAN-and-Stepwise-Model-Growth-Approach-Enhancing-GANs
- We utilize the structure of Auxiliary Classifier GAN (ACGAN) to specify class entries and get additional feedback from the classifier to generate more sophisticated images with less data per class.
- We propose Progressive Step Training (PST), which improves the incremental learning of the PGGAN model. PST solves the stability problem of generating high-resolution images by changing the structure of the generator and discriminator step by step during the learning process. As a result, it improves the classification probability through efficient learning, reduces the learning time and cost, and solves the problem of image distortion and blurring.
- We propose a novel weight adjustment mechanism that reduces the loss of image characteristics during the transition to high resolution and improves the performance of the model.
- We validate the performance of the proposed model through experiments on CIFAR-10 training data, which is used in various research and development in the field of computer vision.

## What problem are we trying to solve?

#### - Image distortion

#### - Unstable learning 


## What workaround do you suggest? 

####  Developing a weight transfer mechanism to utilize previous model weights.

<img src="./images/Linear Mixing for Weighting on a Model-by-Model Basis .png" width="400px"></img>

####  Implementing a model training loop with hyperparameter adjustments based on the learning rate.

<img src="./images/Progressive Generator.png" width="400px"></img>
<img src="./images/Progressive Discriminator.png" width="400px"></img>

## Evaluation 

### Compare accuracy

<img src="./images/Compare accuracy.png" width="400px"></img>

### Compare performance

#### Traditional 

<img src="./images/Average Traditional GAN performance.png" width="400px"></img>

#### PST

<img src="./images/Average PST performance.png" width="400px"></img>

### Generated images on CIFAR10
<img src="./images/Generated images on CIFAR10.png" width="400px"></img>

See the paper for details...
