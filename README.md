# Waste Management Using CNN Model ♻️

## Project Overview
This project implements a Convolutional Neural Network (CNN) to classify waste images into two categories: **Organic** and **Recyclable**. The model is designed to automate waste management by classifying images of waste products efficiently using deep learning.

# Table of Contents
- [Project Overview](#overview)
- [Dataset](#dataset)
- [Technologies & Requirements](#technologies--requirements)
- [Week-wise Work Done](#week-wise-work-done)
- [Usage](#usage)
- [License](#license)
- [Model Performance](#Model-Performance)
- [Accuracy & Loss Graphs](#Accuracy-&-Loss-Graphs)
- [Testing the Model](#Testing-the-Model)
- [Making Predictions](#Making-Predictions)
- [Visualizing Sample Predictions](#Visualizing-Sample-Predictions)
- [Conclusion & Summary](#Conclusion-&-Summary)
- [Future Improvements](#Future-Improvements)
- [Credits](#Credits)
- [License](#License)
- [Acknowledgements](#acknowledgements)
- [Contact](#contact)

---

## Technologies Used  
- Python  
- TensorFlow/Keras  
- OpenCV  
- NumPy  
- Pandas  
- Matplotlib

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
```
tensorflow
keras
numpy
pandas
opencv-python
cv2 (OpenCV)
matplotlib
tqdm
kagglehub
```

# Week-wise Work Done 📅

## Week 1: Data Preprocessing and Visualization
- Data Loading: Loaded images from the training and testing dataset folders.
- Image Preprocessing: Converted the images to RGB format using OpenCV's cv2.cvtColor().
- Data Visualization: Used a pie chart to visualize the distribution of waste categories (Organic and Recyclable).

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
- Model Architecture: Defined a CNN architecture using Conv2D, MaxPooling2D, Flatten, Dense layers to classify images.
- Compilation: Compiled the model with the Adam optimizer and binary cross-entropy loss function.
- Model Training: Trained the model using the ImageDataGenerator class for both the training and testing datasets.
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
## Download the dataset:
Dataset for the project is linked here: ```https://www.kaggle.com/datasets/techsash/waste-classification-data```
## Train the model:
```
hist = model.fit(
    train_generator, 
    epochs=10, 
    validation_data=test_generator
)
```
## Evaluate the model: After training, you can evaluate the model's performance on the test set.

# 📈 Model Performance

The model was evaluated on test data, yielding the following results:

✅ Test Accuracy: 89.61%

✅ Test Loss: 0.3773

This means the model correctly classifies waste in 89 out of 100 cases on unseen data.

# 📊 Accuracy & Loss Graphs

Two key metrics are plotted:

Accuracy Graph: Shows improvement over epochs.

Loss Graph: Helps monitor overfitting.
```
import matplotlib.pyplot as plt

train_acc = hist.history['accuracy']
val_acc = hist.history['val_accuracy']
train_loss = hist.history['loss']
val_loss = hist.history['val_loss']

epochs = range(1, len(train_acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs, train_acc, 'bo-', label='Training Accuracy')
plt.plot(epochs, val_acc, 'ro-', label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Training vs Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs, train_loss, 'bo-', label='Training Loss')
plt.plot(epochs, val_loss, 'ro-', label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training vs Validation Loss')
plt.legend()

plt.show()
```

# 🛠 Testing the Model

The trained model is loaded and evaluated on test images

# 🏷 Making Predictions

The model predicts the class labels of test images

# 🖼️ Visualizing Sample Predictions

Randomly selected images are displayed with predicted labels

# 📌 Conclusion & Summary

## 1️⃣ Overview of the Model

The CNN model classifies waste into Organic (O) and Recyclable (R) categories.

Used convolutional layers, max-pooling, batch normalization, and dense layers.

Optimized using categorical cross-entropy loss and evaluated based on accuracy.

## 2️⃣ Model Evaluation on Test Data

✅ Test Accuracy: 89.61%

✅ Test Loss: 0.3773

🔹 The model performs well on unseen data, with minimal overfitting.

## 3️⃣ Predictions and Sample Results

Predicted vs. Actual Classes:

Predicted Classes: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

Actual Classes: ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O']

# 🎯 Future Improvements

- 🔹 Enhance Dataset: Include more diverse images to improve generalization.
- 🔹 Hyperparameter Tuning: Experiment with different optimizers and learning rates.
- 🔹 Transfer Learning: Use a pre-trained model like ResNet or VGG16 for better feature extraction.

# 📢 Credits

Developed by [Nandini Goyal]

# License 📜

This project is licensed under the MIT License - See the [LICENSE](LICENSE) file for details. 

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
- **LinkedIn**: [https://www.linkedin.com/in/nandini-goyal-6b9116259/](https://www.linkedin.com/in/nandini-goyal-6b9116259/)
- **GitHub**: [https://github.com/NandiniGoyal16](https://github.com/NandiniGoyal16)

I would love to hear from you! 😊















