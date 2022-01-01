# Binary Text Classification

Classification of images into texted and non-texted category using Convulation Neural Network. 

## Table Of Contents

* [About the Project](#about-the-project)
* [Built With](#built-with)
* [Overview](#overview)

## About The Project

This is a CNN model built with Tensorflow for binary classification of images. It is a part of India Academia Connect AI Hackathon. 

Dataset contains Training and testing sets. 

Training dataset folder - NON TEXT images are stored in 'background' sub folder and TEXT images are stored in 'hi' sub folder.

Test dataset folder - Contains both TEXT and NON-TEXT images to be classified with label 1 for TEXT images and label 0 for NON_TEXT images.

## Built With


* [TensorFlow](https://www.tensorflow.org/)
* [Keras](https://keras.io/)
* [Numpy](https://numpy.org/)
* [Pandas](https://pandas.pydata.org/)
* [OpenCV](https://opencv.org/)
* [Matplotlib](https://matplotlib.org/)

## Overview
### Dataset

**Two dataset**: Training and Testing set. 

Training dataset contains two sets of data:
- One with only images.
- Other one, with text written over images.

### Aim
To classifiy the Testing dataset into texted or non-texted i.e Binary classification of the images present inside the testing folder.

### Preprocessing
 ```
 train_data = train.flow_from_directory("training/",target_size=(64, 64), batch_size=32, class_mode='binary')
 ```
 > Found 5875 images belonging to 2 classes.
 
 ```
 train_data.class_indices
 ```
> {'background': 0, 'hi': 1}
```
train_data.classes
```
> array([0, 0, 0, ..., 1, 1, 1])

### Model Training evaluation
```
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32,(3,3), activation='relu', input_shape = (64, 64, 3)), 
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPool2D(2,2),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(64, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
    ])
 model.summary()
 ```
 ![model_summary](https://github.com/Dipankar-Medhi/Binary_text_classification-CNN/blob/master/model_summary.jpg)
``` 
model.compile(loss = 'binary_crossentropy',
optimizer='rmsprop', metrics=['accuracy'])
model_fit = model.fit(train_data, epochs=10, steps_per_epoch=20, validation_data=validation_data, validation_steps=1000)
```
> Epoch 1/10
20/20 [==============================] - ETA: 0s - loss: 0.2214 - accuracy: 0.9203WARNING:tensorflow:Your input ran out of data; 
interrupting training. Make sure that your dataset or generator can generate at least `steps_per_epoch * epochs` batches (in this case, 1000 batches). 
You may need to use the repeat() function when building your dataset.
20/20 [==============================] - 10s 500ms/step - loss: 0.2214 - accuracy: 0.9203 - val_loss: 0.2136 - val_accuracy: 0.9253
Epoch 2/10
20/20 [==============================] - 2s 82ms/step - loss: 0.2506 - accuracy: 0.9094
Epoch 3/10
20/20 [==============================] - 2s 83ms/step - loss: 0.2406 - accuracy: 0.9109
Epoch 4/10
20/20 [==============================] - 2s 84ms/step - loss: 0.2171 - accuracy: 0.9203
Epoch 5/10
20/20 [==============================] - 2s 87ms/step - loss: 0.2172 - accuracy: 0.9250
Epoch 6/10
20/20 [==============================] - 2s 84ms/step - loss: 0.2354 - accuracy: 0.9000
Epoch 7/10
20/20 [==============================] - 2s 82ms/step - loss: 0.2322 - accuracy: 0.9171
Epoch 8/10
20/20 [==============================] - 2s 85ms/step - loss: 0.1889 - accuracy: 0.9234
Epoch 9/10
20/20 [==============================] - 2s 81ms/step - loss: 0.2154 - accuracy: 0.9234
Epoch 10/10
20/20 [==============================] - 2s 85ms/step - loss: 0.2191 - accuracy: 0.9234


```
verdict = model.predict(test_img)
verdict[0][0]
```
> 1.0



Model             |  Loss Function
:-------------------------:|:-------------------------:
<img align="left" width="400" height="800" alt='model' src="https://github.com/Hello-Peter-GPU-Hackathon/CNN-Image-Classification/blob/main/model.png">  |  <img width = '600' height = '400' alt = 'loss function' src = 'https://github.com/Hello-Peter-GPU-Hackathon/CNN-Image-Classification/blob/main/books_read.png' >
A graphical representation of the CNN model. | A graphical representation of the loss function.




