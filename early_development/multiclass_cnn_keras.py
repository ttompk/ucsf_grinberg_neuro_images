'''
#Train a simple deep CNN on the CIFAR10 small images dataset.
It gets to 75% validation accuracy in 25 epochs, and 79% after 50 epochs.
(it's still underfitting at that point, though).


The keras methodology:
Split data into train/test
Pick a keras model - sequential is the basic architecture
Add layers as needed:
    Note: in keras the first layer requires more parameters than subsequent layers. These 
        parameters are passed through via the model class (e.g. sequential)
    Weight initialization:   (fan_in: n of input units, fan_out: n of output units)
            'glorot_uniform' - Xavier uniform. Draws samples from a uniform distribution 
                within [-limit, limit] where limit is sqrt(6 / (fan_in + fan_out)) where 
                fan_in is the number of input units in the weight tensor and fan_out is 
                the number of output units in the weight tensor.
            'glorot_normal' - Xavier normal. Draws samples from a truncated normal 
                distribution centered on 0 with stddev = sqrt(2 / (fan_in + fan_out)).
            'he_normal' - draws samples from a truncated normal distribution centered 
                on 0 with stddev = sqrt(2 / fan_in)
            'he_uniform' - draws samples from a uniform distribution within [-limit, limit] 
                where limit is sqrt(6 / fan_in).
            ... others (https://keras.io/initializers/)
    Types of layers in keras:
        dense - simple layer. output = activation(dot(input, kernel) + bias)
        activation - applies activation function to output (e.g. ReLu)
        dropout - prevents overfitting by randomly dropping units (along with connections) 
            during training. Better than regularization (so says this
            paper: http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf)
        flatten - converts input to one dimension for analysis in output layer
        
Use the '.compile' method to tell it how to learn (loss, optimizer, metrics)
    Loss:  'mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error',
        'mean_squared_logarithmic_error', 'squared_hinge', 'hinge', 'categorical_hinge', 'logcosh',
        'categorical_crossentropy', 'sparse_categorical_crossentropy', 'binary_crossentropy',
        'kullback_leibler_divergence', 'poisson', 'cosine_proximity'
    Optimizers:  'SGD', 'RMSprop', 'Adagrad', 'Adam', 'Adamax', 'Nadam'
    Metrics:  'binary_accuracy', 'categorical_accuracy', 'sparse_categorical_accuracy',
        'top_k_categorical_accuracy', 'sparse_top_k_categorical_accuracy'
Fit the the data using batches
Evaluate model performance on training data
generate predictions on unseen data


'''

#from __future__ import print_function
import os
import keras
from keras.datasets import cifar10

# image preprocessing class
from keras.preprocessing.image import ImageDataGenerator

# The nn architecture. The sequential model is the basic/classic 'linear' stack of layers. 
from keras.models import Sequential  

# basic hidden layers 
from keras.layers import Dense  
from keras.layers import Dropout 
from keras.layers import Activation
from keras.layers import Flatten

# image-related hidden layers - typically convolusion and pooling
from keras.layers import Conv2D
from keras.layers import MaxPooling2D

# define structure of nn
batch_size = 32
num_classes = 10
epochs = 10  # start with 10, move to 100 (in this example)
data_augmentation = True  # attribute to the ImageDataGenerator class. Augmentation
num_predictions = 20  # uh?
save_dir = os.path.join(os.getcwd(), 'saved_models')   # cool!
model_name = 'keras_cifar10_trained_model.h5'

# The data, split between train and test sets:
(x_train, y_train), (x_test, y_test) = cifar10.load_data()
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Convert class vectors to binary class matrices. This is done because...
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# initialize basic model
model = Sequential()

# module 1 - 32 dimension output
# convo
m1_dimensions=32
m1_kernel_size=(3,3)
strides=(1,1)
padding='same'
kernel_init = 'glorot_uniform'   # weight initialization

model.add(Conv2D(filters=m1_dimensions, kernel_size=m1_kernel_size, strides=strides, padding=padding,
                 kernel_initializer=kernel_init,
                 input_shape=x_train.shape[1:]))  #note: 'input_shape' only needed if first layer of model
model.add(Activation('relu')) # activation for the previous layer.
model.add(Conv2D(m1_dimensions, m1_kernel_size))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))  # downscale data factor - can stretch i.e. (2,3)
model.add(Dropout(0.25))

# module 2 - 64 dimension output
m2_dimensions=64
m2_kernel_size=(3,3)
strides=(1,1)
padding='same' 
model.add(Conv2D(m2_dimensions, m2_kernel_size, padding=padding))
model.add(Activation('relu'))
model.add(Conv2D(m2_dimensions, m2_kernel_size)
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# output layer
model.add(Flatten())
model.add(Dense(units=512)) # units is the dimensionalty of the output space
model.add(Activation('relu')) 
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# initiate RMSprop optimizer
opt = keras.optimizers.RMSprop(lr=0.0001, decay=1e-6)
#opt = optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)

# Let's train the model using RMSprop 
model.compile(loss='categorical_crossentropy',
              optimizer=opt,
              metrics=['accuracy'])

x_train = x_train.astype('float32')  # why change to float32? Save space?
x_test = x_test.astype('float32')    
x_train /= 255
x_test /= 255

if not data_augmentation:
    print('Not using data augmentation.')
    model.fit(x_train, y_train,
              batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True)
else:
    print('Using real-time data augmentation.')
    # This will do preprocessing and realtime data augmentation:
    datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        zca_epsilon=1e-06,  # epsilon for ZCA whitening
        rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
        # randomly shift images horizontally (fraction of total width)
        width_shift_range=0.1,
        # randomly shift images vertically (fraction of total height)
        height_shift_range=0.1,
        shear_range=0.,  # set range for random shear
        zoom_range=0.,  # set range for random zoom
        channel_shift_range=0.,  # set range for random channel shifts
        # set mode for filling points outside the input boundaries
        fill_mode='nearest',
        cval=0.,  # value used for fill_mode = "constant"
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False,  # randomly flip images
        # set rescaling factor (applied before any other transformation)
        rescale=None,
        # set function that will be applied on each input
        preprocessing_function=None,
        # image data format, either "channels_first" or "channels_last"
        data_format=None,
        # fraction of images reserved for validation (strictly between 0 and 1)
        validation_split=0.0)

    # Compute quantities required for feature-wise normalization
    # (std, mean, and principal components if ZCA whitening is applied).
    datagen.fit(x_train)

    # Fit the model on the batches generated by datagen.flow().
    model.fit_generator(datagen.flow(x_train, y_train,
                                     batch_size=batch_size),
                        epochs=epochs,
                        validation_data=(x_test, y_test),
                        workers=4)

# Save model and weights
if not os.path.isdir(save_dir):
    os.makedirs(save_dir)
model_path = os.path.join(save_dir, model_name)
model.save(model_path)
print('Saved trained model at %s ' % model_path)

# Score trained model.
scores = model.evaluate(x_test, y_test, verbose=1)
print('Test loss:', scores[0])
print('Test accuracy:', scores[1])