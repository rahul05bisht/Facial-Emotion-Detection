# **Facial Emotion Detection Model**  

This project is a **Facial Emotion Detection System** that classifies human emotions into seven categories using a **deep learning model** trained on the **FER2013 dataset** from Kaggle.  

---

## 📌 **Features**  
- 🎥 **Detects emotions** from real-time **webcam feed** or **images**  
- 📊 **Trained on FER2013** (Facial Expression Recognition 2013) dataset  
- 🧠 **Uses CNN (Convolutional Neural Networks)** for classification  
- ⚡ **Implemented with Keras, OpenCV, and TensorFlow**  

---

## 🗂 **Dataset**  
- The model is trained using the **FER2013 dataset**, which contains **35,887 grayscale images (48x48 pixels)** categorized into:  
  - 😠 **Angry**  
  - 🤢 **Disgust**  
  - 😨 **Fear**  
  - 😃 **Happy**  
  - 😐 **Neutral**  
  - 😢 **Sad**  
  - 😲 **Surprise**  

🔗 **Dataset Source:** [FER2013 on Kaggle](https://www.kaggle.com/datasets/msambare/fer2013)  

---

## ⚙️ **Model Training**  
- 🏗 **CNN model with 4 convolutional layers**  
- 🔹 **Batch size:** `32`  
- 🚀 **Optimizer:** `Adam`  
- 🔥 **Loss function:** `Categorical Crossentropy`  
- 🕒 **Training duration:** `50 epochs`  

---

## 🚀 **How to Run**  

### 🔹 **1️⃣ Install Dependencies**  
Run the following command to install required libraries:  
```pip install tensorflow keras numpy opencv-python matplotlib seaborn
```

**Run Emotion Detection on Images**
-	Run this file: emotion_detect_from_image.py 
-	As 2-3 sample images were in img folder you can also add to that and check for custom image just change image name with extension in line 14
frame=cv2.imread("img/surp.jpeg") 
**Run Emotion Detection on Webcam**
- Run this file: real_time_emotion.py
**To Train model and work with your model**
-	Download Dataset from Kaggle FER2013 and then check path 
-	Check that the dataset paths in your code (train_data_dir and validation_data_dir) are correctly set to your local directory.
-	The default settings in this project are:
 **batch_size = 32**
 **epochs = 50**
-	You can adjust batch size and epochs based on your system’s capacity:
**Higher batch size** (e.g., 64 or 128) → Faster training but requires more RAM
**Lower batch size** (e.g., 16) → Slower training but works on low-memory systems
**More epochs** (e.g., 150+) → Improves accuracy but increases training time
-	Run file in terminal  it may take some time from hours to day after that one file will be saved to folder with extension .h5 
-	Once training is done, you can use the trained model for real-time emotion detection via webcam  and for image also ,just use your model file name in here:
model=load_model('src/model_file_50.h5')

