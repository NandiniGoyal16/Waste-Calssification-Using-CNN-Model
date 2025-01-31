# Waste Management Using CNN Model ♻️

## Overview
This project implements a Convolutional Neural Network (CNN) to classify waste images into two categories: **Organic** and **Recyclable**. The model is designed to automate waste management by classifying images of waste products efficiently using deep learning.

# Table of Contents
- [Project Overview](#overview)
- [Dataset](#dataset)
- [Technologies & Requirements](#technologies--requirements)
- [Week-wise Work Done](#week-wise-work-done)
- [Usage](#usage)
- [License](#license)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Dataset 📂
The dataset for the project is linked here: [Waste Classification Dataset](https://www.kaggle.com/datasets/techsash/waste-classification-data).

To download the dataset directly, you can use the following code:

```python
import kagglehub # you may have to pip install that!
path = kagglehub.dataset_download("techsash/waste-classification-data")
print("Path to dataset files:", path)
```
# Technologies & Requirements ⚙️

Libraries Used:
tensorflow
keras
numpy
pandas
cv2 (OpenCV)
matplotlib
tqdm
Python Dependencies:
Install the required libraries by running:
```
pip install -r requirements.txt
```
requirements.txt:
```
tensorflow
numpy
pandas
opencv-python
matplotlib
tqdm
kagglehub
```
# Week-wise Work Done 📅

## Week 1: Data Preprocessing and Visualization
Data Loading: Loaded images from the training and testing dataset folders.
Image Preprocessing: Converted the images to RGB format using OpenCV's cv2.cvtColor().
Data Visualization: Used a pie chart to visualize the distribution of waste categories (Organic and Recyclable).

```
x_data = []
y_data = []
for category in glob(train_path+'/*'):
    for file in tqdm(glob(category+'/*')):
        img_array = cv2.imread(file)
        img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        x_data.append(img_array)
        y_data.append(category.split('/')[-1])

# Visualizing the class distribution
plt.pie(data.label.value_counts(), labels=['Organic', 'Recyclable'], autopct='%0.2f%%', 
        colors=['#a0d157', '#c48bb8'], startangle=90, explode=[0.05, 0.05])
plt.show()
```
## Week 2: Model Architecture & Training
Model Architecture: Defined a CNN architecture using Conv2D, MaxPooling2D, Flatten, Dense layers to classify images.
Compilation: Compiled the model with the Adam optimizer and binary cross-entropy loss function.
Model Training: Trained the model using the ImageDataGenerator class for both the training and testing datasets.
```
model = Sequential()

# First Convolutional Layer
model.add(Conv2D(32, (3, 3), input_shape=(224, 224, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Second Convolutional Layer
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Third Convolutional Layer
model.add(Conv2D(128, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D())

# Flatten the output
model.add(Flatten())

# Dense Layers
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))

# Output Layer
model.add(Dense(2))
model.add(Activation('sigmoid'))

# Compiling the model
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])

# Summary of the model architecture
model.summary()
```
# Usage 🚀

## Clone the repository:
```git clone https://github.com/yourusername/waste-management-cnn.git```
## Install dependencies:
```pip install -r requirements.txt```
## Download the dataset:
```import kagglehub # you may have to pip install that!
path = kagglehub.dataset_download("techsash/waste-classification-data")
print("Path to dataset files:", path)
```
## Train the model:
```
hist = model.fit(
    train_generator, 
    epochs=10, 
    validation_data=test_generator
)
```
## Evaluate the model: After training, you can evaluate the model's performance on the test set.

# License 📜

This project is licensed under the MIT License - see the LICENSE file for details.

# Acknowledgments 🎉

Dataset Source: Kaggle - Waste Classification Dataset
OpenCV: Used for image preprocessing and visualization.
TensorFlow/Keras: Used to build and train the CNN model.

### Explanation of the Sections:
- **Overview**: A brief description of the project.
- **Dataset**: Information about where to get the dataset, along with a Python code snippet for downloading it programmatically.
- **Technologies & Requirements**: Lists the necessary libraries and dependencies, as well as instructions to install them via `pip`.
- **Week-wise Work Done**: Provides a breakdown of tasks completed week-wise, with code snippets for reference.
- **Usage**: Instructions on how to clone the repo, install dependencies, and run the model.
- **License**: Adds a standard MIT License section, which you can customize or change based on your preference.
- **Acknowledgments**: Credit for dataset and libraries used.

This structure will make your GitHub repository clear and easy to follow for anyone interested in your project. Let me know if you need any adjustments!

# Contact 📬

If you have any questions or suggestions, feel free to reach out!

- **Email**: [nandini04.goyal@gmail.com](mailto:nandini04.goyal@gmail.com)
- **LinkedIn**: [LinkedIn Profile](https://www.linkedin.com/in/your-linkedin)
- **GitHub**: [Your GitHub Profile](https://github.com/yourusername)

I would love to hear from you! 😊















