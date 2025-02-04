# Brain Tumor Classification 

<p align="center">
  <img src="./images/brain_tumor_classification.jpg">
</p>



## Problem Statement
A brain tumor is a mass of abnormal cells that grows in the brain, it can be benign (non-cancerous) or malignant (cancerous). Its symptoms are:

- Headaches, which can be severe, persistent, or come and go
- Seizures, which can be mild or severe
- Weakness or paralysis in part of the body
- Loss of balance
- Changes in mood or personality
- Changes in vision, hearing, smell, or taste
- Nausea and vomiting
- Difficulty speaking
- Difficulty swallowing

The growth of brain tumors can cause the pressure inside the skull to increase leading to brain damage, and loss of life if not discovered early and properly treated.

This project is aimed at developing a robust brain tumor detection model using Convolutional Neural Networks (CNNs) to automate the analysis of magnetic resonance imaging (MRI) scan by accurately identifying and classifying the tumors at an early stage which can reduce the load on doctors, help in selecting the most convenient treatment method and hence increase the rate of survival.

This project was implemented as a requirement for the completion of [Machine Learning Zoomcamp](https://github.com/DataTalksClub/machine-learning-zoomcamp) - a free course about Machine Learning.



## Exploratory Data Analysis
The dataset used in this project is the [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset) from kaggle. 

It consists of 7,023 magnetic resonance imaging scans, annotated in a folder structure of 5,712 test and 1,311 train. The dataset consists of images of no tumour and brain tumor types: pituitary, meningioma, and glioma. The diagrams below show the histogram of the class distribution of training and testing data.

<p align="center">
  <img src="./images/train_data.jpg">
</p>

<p align="center">
  <img src="./images/test_data.jpg">
</p>

Finally, some samples of the dataset were drawn as shown below:

<p align="center">
  <img src="./images/sample_images.jpg">
</p>



## First trainings
After conducting the exploratory data analysis, 3 pre-trained deep convolutional neural network models were used to train the data for 10 epochs each with the aim of finding the one that will return the best test accuracy which will then be used to create the final model. The results obtained with BATCH_SIZE = 32 and LEARNING_RATE = 0.001, are shown below:

Xception - 98.93%
InceptionV3 - 98.32%
ResNet101V2 - 96.80%

Xception was selected because of its higher test accuracy

The plot of the run of each model are shown below:

### Xception
<p align="center">
  <img src="./images/Xception_result.jpg">
</p>

### InceptionV3
<p align="center">
  <img src="./images/InceptionV3_result.jpg">
</p>

### ResNet101V2
<p align="center">
  <img src="./images/ResNet101V_result.jpg">
</p>



## Final training
The final training was conducted using all the tuning hyper-parameters and the results obtained are listed below:

InceptionV3 with BATCH_SIZE = 32, DROP_RATE = 0.5, LEARNING_RATE = 0.00001
Mean Train Accuracy: 0.715
Mean Test Accuracy: 0.6725
Mean Train Loss: 0.5475
Mean Test Loss: 0.6174

As expected, the result was much higher, and this can be attributed to the reasons below:

Firstly, smaller batch sizes can lead to more frequent updates, which may improve convergence speed thereby providing better accuracy, but can be computationally expensive and time-consuming. It requires less memory and can lead to better generalization due to less noise during the optimization process.

Secondly, dropout prevents the network from becoming too dependent on specific neurons, which improves the model's ability to generalize to new data, it forces the network to learn more generalized representations by preventing it from relying on specific features, it can be seen as training an ensemble of smaller networks, which helps the model learn more robust features.

Finally, the right learning rate will enable the model to converge on something useful while still training in a reasonable amount of time, this can be achieved by cyclical learning rate that involves letting the learning rate oscillate up and down during training, exponential decay that decreases the learning rate at an exponential rate, which can lead to faster convergence, and ReduceLROnPlateau that adjusts the learning rate when a plateau in model performance is detected. The plot of the run is shown below:

<p align="center">
  <img src="./images/InceptionV3_final.jpg">
</p>



## The model
The resultant model obtained from the final model was saved in keras format so that it can be hosted online for inferencing. The size of the model on disk is 274 MB which makes it impossible to host directly online and the solution to this problem is to enlist the use of tools that will help in solving this problem. Docker and kubernetes are the defacto tools used to solve this problem. 

Docker is a software platform that makes it possible to build, test, and deploy applications quickly. Docker packages software into standardized units called containers that provides all a software needs to run including libraries, system tools, code, and runtime. Using Docker enables quick deployment and scaling of applications into any environment and with a guarantee of the code running.

Kubernetes is an open-source software that allows the deployment and management of containerized applications at scale. Kubernetes manages clusters of cloud compute instances and runs containers on those instances with processes for deployment, maintenance, and scaling. Using Kubernetes, enables running any type of containerized applications using the same toolset on-premises and in the cloud.

The advantages of using docker and kubernetes together are scalability, high availability, portability, security, ease of use, reduced costs, improved agility. The image below shows the aggregated summary of the model.

 <p align="center">
  <img src="./images/InceptionV3_params.jpg">
</p>



## The project
After saving the model and deciciding on the tools for inferencing, the next task is to built the project by 
starting with the virtual environment followed by the required files in the sequence shown below:

> pip install pipenv
> pipenv install tensorflow==2.16.2
> pipenv install numpy==1.26.4
> pipenv install pillow flask gunicorn requests keras_image_helper
> pipenv install --dev notebook==7.2.1 ipywidgets jupyter pandas matplotlib seaborn

Then 