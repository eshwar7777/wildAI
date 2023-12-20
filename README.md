# wildAI
an automated AI and IOT-driven solution for elephant-human conflict

ABSTRACT
Human-wildlife conflict has always been a major problem leading to casualties and loss of property. But this can very well be mitigated by proper monitoring and alerting at recognized animal passes and crossings. Roadkill is a serious threat to most of the endangered species. Crossing elephants and other large mammals can create life-threatening hazards on both road and rail. It takes a toll on the lives of humans and wildlife.
In this paper, we hereby propose an automated AI and IoT-driven solution (WildAI) to avoid the said conflict in the context of animal conservation. In our proposed system we have taken an elephant as a specimen under test. After successful recognition of the animal, in the forest border areas, the signal is passed on to a base station using LoRaWAN Protocol which then transmits it further towards designated check posts near elephant corridors.

Chapter 1: Introduction

1.1 Problem Statement

Our aim here is to curb the casualties and loss of property at dedicated elephant corridors. We use the latest advances in technology such as Artificial Intelligence (AI) and Internet of Things (IoT) to create an early messaging system that can address the aforementioned challenges. 

1.2 Motivation
Our motive is to save the lives of wild animals from accidents by using
the latest technologies available in the present day. If we make this one
most in the utmost accuracy, we can implement this on several other domains
also.

Figure 1. An elephant herd crossing through a railway track. [Biplab Hazra/Quartz India]

Figure 2 Railway tracks often cut through elephant corridors, causing fatality. [Biplab Hazra/Quartz India]


Figure 3. [Source: DECCAN CHRONICLE]


1.3 Objectives
The following are the objectives of our project:


The main objective of our proposed system is to prevent the wild animals from accidents and to notify the base station.
This approach believes that by using various technologies such as Machine Learning and the Internet of Things the environmental balance can be achieved by saving the wild animals from getting harmed.
The proposed method is also easy to implement and environmentally friendly. It can save wildlife and property.
The proposed solution for the wildlife alert system presents a cost-effective, reliable, and technically simple solution.
The project takes the approach of harmonious coexistence, by ensuring that both- humans and wildlife- are protected from conflict.



Chapter 2: Literature Survey

Background details of different concepts used in your project along with related research work

2.1 Background Details

Study of different concepts
Collecting the datasets for our project from Kaggle.
Uploading the collected datasets in Google Drive.
All functions for running the code in Google Colab.
Mounting the datasets from Google Drive to Google Colab by setting the path of datasets.
All the concepts (Functions, Methods, Keywords, Syntax, etc.) of Python for writing the Machine learning code.
All the concepts of TensorFlow Library to create the Deep Learning Model for our projects.
All the concepts of Neural Network of Machine Learning.
All concepts of Convolutional Neural Network (CNN) for our Deep Learning model.
Saving the trained model in Google Drive.
Loading the trained model in Google Colab for testing it on different datasets.
Updating the Raspberry Pi3 to use it efficiently to detect objects by using its camera module and produce the alert system.
Loading the trained model in Google Colab on Raspberry Pi3.
Detecting the objects by using the camera module of Raspberry Pi3.
All concepts of the Internet of Things for producing the alert using LoRaWAN (Sender and Receiver)  IoT device.





Chapter 3: Proposed System
Overall working principle of our project


3.1 Proposed Methodology

We have deployed a trained CNN model (currently) specialized in the willing detection of a particular animal species, which would be processing the images captured from a compatible camera module (Raspberry Pi3 camera module) after fixed intervals. The CNN result from the classification problem (if yes) would be conveyed to the interface (between ML & IoT modules). The interface then would handle communication between the ML and IoT modules. 
Once this is done the IoT modules trigger the Sender to send signals over the LoRaWAN protocol. The Receiver located at a distance upon receiving these signals produces the alert.

3.2 System Architecture


Figure 4 SYSTEM ARCHITECTURE OF OUR PROJECT

3.3 Flow Chart /Algorithm


3.4 Hardware Components

Different hardware components we used in the implementation of our project

RaspberryPi3, RPi3 Camera Module (Rpi3 camera Module), LED indicator, and LoRaWAN IoT devices
3.5 Software Tools and Libraries

Different software tools we used in the implementation of our project


Google Drive, Google Colab, Kaggle and Tensorflow 

3.6 Dataset
https://drive.google.com/drive/folders/1dn8WjKr_wKX0xEJjOVus-j_zfMP96qec

The dataset consists of 22930 images with sub-directories Elephant and NotElephant with 11400 and 11531 respectively.

3.7 Evaluation Measure 


Our focus here would be to train the CNN model such that it recognizes the subject animal species in a substantial manner. And gradually would bring it down so that it runs effectively on smaller devices like Raspberry Pi3. Besides this, we aim to develop a bridging interface between Machine Learning and the Internet of Things parts.

Chapter 4: Implementation and Results
We used a dataset consisting of 22930 images with sub-directories Elephant and NotElephant with 11400 and 11531 respectively. Some of the files in the validation folder were not in the format accepted by Tensorflow (JPEG, PNG, GIF, BMP), or may be corrupted. We found the culprit using the imghdr module from the Python standard library, and a simple loop.
TensorFlow is an open-source library for numerical computation that makes ML and developing neural networks efficient and effective. Keras is a high-level API of TensorFlow 2: a highly-productive interface for solving ML problems, with a focus on modern deep learning. It provides required abstractions and building blocks for developing and deploying ML solutions with high iteration velocity.

We defined, trained, visualized, saved, and loaded the CNN model. The model was then loaded into RPi3 after enabling a bridging API for CNN and IoT.

Convolutional neural networks (CNN) are specially built algorithms designed to work with images. It is the process that applies a weight-based filter across every element of an image, helping the computer to understand and react to elements within the picture itself. 
CNN can be helpful when you need to scan a high volume of images for a specific item or feature.

Figure 5 [Source: Flatiron School]

4.1 Implementation Details
80% of the images used for training and 20% for validation
Out of 22931 files belonging to 2 classes, 18345 files were used for training and the rest 4586 files were used for training.
CNN was used for training of model
Various libraries were used for the training and execution of the model:
TensorFlow
TensorFlow.keras
might
path lib

4.2   Results and Analysis

The model summary was obtained as follows:

Model: "sequential"
_________________________________________________________________
 Layer (type)                Output Shape              Param #   
=================================================================
 sequential_1 (Sequential)   (None, 180, 180, 3)       0         
                                                                 
 rescaling_2 (Rescaling)     (None, 180, 180, 3)       0         
                                                                 
 conv2d_3 (Conv2D)           (None, 180, 180, 16)      448       
                                                                 
 max_pooling2d_3 (MaxPooling  (None, 90, 90, 16)       0         
 2D)                                                             
                                                                 
 conv2d_4 (Conv2D)           (None, 90, 90, 32)        4640      
                                                                 
 max_pooling2d_4 (MaxPooling  (None, 45, 45, 32)       0         
 2D)                                                             
                                                                 
 conv2d_5 (Conv2D)           (None, 45, 45, 64)        18496     
                                                                 
 max_pooling2d_5 (MaxPooling  (None, 22, 22, 64)       0         
 2D)                                                             
                                                                 
 dropout (Dropout)           (None, 22, 22, 64)        0         
                                                                 
 flatten_1 (Flatten)         (None, 30976)             0         
                                                                 
 dense_2 (Dense)             (None, 128)               3965056   
                                                                 
 dense_3 (Dense)             (None, 5)                 645       
                                                                 
=================================================================
Total params: 3,989,285
Trainable params: 3,989,285
Non-trainable params: 0
_________________________________________________________________

Restored model was found to be accuracy: 85.10%


Upon visualization the following graphs were obtained:



The above plots show that training accuracy and validation accuracy are off by large margins, and the model has achieved only about 60% accuracy on the validation set.



Installation of tensorflowlite  tools for loading the trained CNN model on rasberrypi:






Captured image of elephant by camera module of rasberrypi:



4.3 Discussion
The basic execution of the CNN code was performed using Google Colab. But as we tried
Implementing with larger dataset, the RAM collapsed, even while the runtime type was set to GPU.
We resolved this by processing the same data in chunks i.e. if the total size of data be 8GB. Then we can divide it into chunks (file) of 1Gb.
12GB data = 12 chunks (files)of 1 GB.
Building the bridging API between CNN and IoT was a challenging task.
LoRaWAN can be used for applications requiring low data rate i.e. upto about 27 kbps.
LoRaWAN network size is limited.
LoRaWAN is not ideal for the real time.

Chapter 5: Conclusion and Scope
In this section, we conclude our project and state scope for further work in future.

5.1 Conclusion


A concept which is applicable for early warning system to reduce human elephant conflict is simulated and the results are obtained and this gives good results. The animals, many of which are already threatened and endangered, are often killed in retaliation. Therefore, To protect the elephant Raspberry Pi3 camera module which is used for real-time wild animal monitoring of elephant and A IoT based sensor (LoRaWAN Device) which is used to produce the alert for the people living in that area to prevent the elephant from getting killed. This Technology is used so that the implementation is simple and less complex. The idea which is explained here helps the forest area to drive away the elephant into the forests and hence avoid from killed. The system is stand-alone and can work for different occasions with less complexity and improved technology by modifying system components and the detection model to fit in different situations, such as tracking livestock and protecting houses against theft.

5.2 Future Scope

Our project can be improved in future to be used in different application following areas:
The current functionalities of our system can be extended and investigate the chance of incorporating its features to other sectors.
We can also make IoT based repelling and monitoring system for crop protection against animal attacks by using the methodology of our system.
We will carry out both hardware and software in a critical order to further improve the effectiveness of our monitoring and alerting system.
The IoT functionalities of our system can be extended from LoRaWAN to our mobile system so that the people beyond the range of alerting system can also get alert message.
The proposed system can be compressed to Tiny ML in order to reduce its complexity and cost.
