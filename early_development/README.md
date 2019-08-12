# Image Detection with Convolutional Neural Net

The purpose of this project is to design and train a neural net to correctly classify images from the [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html) image dataset, [wiki](https://en.wikipedia.org/wiki/CIFAR-10). 

### Data
Location: The data is stored locally. 
N images: 

## Detection algorithm methodology:

Overview:

preprocessing -> [ conv filter -> activation -> pooling ] x 3

### Preprocessing
Method applied: Mean centered

Images are pre-processed using using mean centering or normalization.  

### Architecture
#### Input layer
all data

#### Hidden layer
depth:1
n neurons: 50
activation function: ReLu

#### Output layer
10 classes:  airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, trucks

### Weight Initialization
He Normal (for use with ReLu) 
stddev = sqrt(2 / fan_in)
fan_in: number of incoming inputsls


### Batch Normalization


