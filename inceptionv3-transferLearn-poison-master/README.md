# OneShotKillPoisonAttack

This is a one shot kill attack on transfer learning experiment for the binary classification (dog-vs-fish) that uses inception v3 based on what is described in the [Poison Frogs! paper](https://arxiv.org/abs/1804.00792).

We should define three terms for this attack:
a) base instance: the image that the poison looks like in image/input space
b) poison instance: the instance that gets added to the training data to spur misclassification. Looks like the base instance in input space but like the target instance in feature space.
c) target instance: the instance that we are attacking. We are interested in making this instance be misclassified as the class of the base instance.

The attack works by making a poison instance starting from and staying close to the input representation of a base instance that the poison instance's feature representation is close to the target instance's feature representation. Given a target instance, there are many ways that one could select a base instance. We can select a base instance that its feature representation is close to the target instance feature representation, or we can pick a random base instance, or pick any one that we like. This choice my change the number of iterations needed for doing the optimization. In our experiments, we noticed that the size of the base instance is important and can help in ``making'' the poison in less iterations and therefore have selected some preferred ones which their indices are available in the main_oneShot.py script.

To reproduce the results, the following steps should be taken:

## One-time setup steps
1. Download and extract the pre-trained [Inception v3](http://download.tensorflow.org/models/image/imagenet/inception-2015-12-05.tgz) model and save it in ./inceptionModel (the pb file should be in './inceptionModel/inception-2015-12-05/classify_image_graph_def.pb')
2. Download the images from [imageNet](http://www.image-net.org/) and save the raw images in /Data/rawImages/nameOfCategory [nameOfCategory for dog is: "main_dog"; for fish it is: "fish"]. The synset id for the dogs is: n02084071 and for the fish is: n02512053

## Running the main file for making poisons: main_oneShot.py
Update directories for the saved raw images as needed. It is initially set to main_dog and fish.

If this is the first time running the script, set firsTime = True in the main_oneShot.py script. This will do the following:
1. load the images into numpy arrays
2. removes images that might cause an issue with inception - these are the ones which have less than 3 dimensions
3. saves the numpy ndarrays for future use
4. run the images through inception and get the feature representations for the images - this step may be time consuming since incpetion does not allow insertion of a batch of images to the 'DecodeJpeg:0' tensor. But don't worry, we will save them for re-using -- its a one time thing.
5. Load the bottleneck tensors and do the train test split and save it

If it is not the first time running the script and all of the data is loaded, the remainder of script does the following:
1. load the train and test data 
2. treat every test image as a target instance and depending on the parameters, select a base instance from the opposite class.
3. make the poison instance and save the numpy array and the "compressed JPEG" version of it for visualizing it. 

``Poison making'' is done via a Forward-Backward-Splitting Algorithm. For details see the utility file and/or view our paper.

## Checking the effectiveness of the one-shot-kill attacks on the poisoned datasets: coldStartPerformance.py or warmStartPerformance.py

In this part, we check the effectiveness for every attack. Since we are going to be re-using the same graph for every attack, we save the graphdefs to file. This step is a one time step, if you have already done this, no need to re-do it. But if you haven't, you should run make_graph_fr_warm_and_cold.py

Then depending on the attack being warm start or cold start, we should run the appropriate file.


If you find this study useful for your own research, please consider citing the Poison Frogs paper:
```
@ARTICLE{2018arXiv180400792S,
   author = {{Shafahi}, A. and {Ronny Huang}, W. and {Najibi}, M. and {Suciu}, O. and 
	{Studer}, C. and {Dumitras}, T. and {Goldstein}, T.},
    title = "{Poison Frogs! Targeted Clean-Label Poisoning Attacks on Neural Networks}",
  journal = {ArXiv e-prints},
archivePrefix = "arXiv",
   eprint = {1804.00792},
 primaryClass = "cs.LG",
 keywords = {Computer Science - Learning, Computer Science - Cryptography and Security, Computer Science - Computer Vision and Pattern Recognition, Statistics - Machine Learning},
     year = 2018,
    month = apr,
   adsurl = {http://adsabs.harvard.edu/abs/2018arXiv180400792S},
  adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```

dataset : [CIFAR 10](https://www.cs.toronto.edu/~kriz/cifar.html)
