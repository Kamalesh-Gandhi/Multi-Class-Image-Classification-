# 🐟 Multiclass Fish Image Classification using Deep Learning                              
 
A robust deep learning solution to classify different species of fish using Convolutional Neural Networks and Transfer Learning, deployed via a Dockerized Streamlit app on AWS EC2.

## 🧠 Problem Statement

Accurate species identification plays a vital role in marine biology, ecological monitoring, and sustainable fisheries management. However, manual classification of fish species is time-consuming, error-prone, and requires expert knowledge—especially when species have subtle visual differences.

With the rise in available fish imagery from underwater surveys, fisheries, and environmental databases, there is a growing need for an automated solution that can identify fish species efficiently and reliably from images.

This project addresses the challenge by developing a deep learning-based image classification system that can accurately classify multiple species of fish using Convolutional Neural Networks (CNNs) and pre-trained transfer learning models. The final solution is optimized for deployment via a web interface to ensure accessibility and ease of use for researchers, conservationists, and industry stakeholders.

## 📊 Dataset Description

- The dataset consists of images of fish, organized into separate folders based on their species.
- It includes **training**, **validation**, and **test** sets for effective model evaluation.
- Each class folder contains labeled images that serve as inputs for training Convolutional Neural Networks.
- The dataset was loaded and preprocessed using **TensorFlow's `ImageDataGenerator`**, enabling efficient real-time data augmentation.
- Targeted augmentation techniques were applied specifically to handle **class imbalance**, especially in underrepresented classes like `animal_fish_bass`.

📦 **Download the dataset**:  
[Click here to access the dataset on Google Drive](https://drive.google.com/drive/folders/1iKdOs4slf3XvNWkeSfsszhPRggfJ2qEd)


## 🧠 Model Architectures

The project implements a combination of custom-built and pre-trained deep learning models to perform multiclass fish image classification. Both types of models were evaluated to identify the most effective architecture for the task.

---

### 🧱 1. Custom CNN (Built from Scratch)

A deep Convolutional Neural Network was built using multiple convolutional blocks with the following characteristics:
- 3 Convolutional Blocks with:
  - `Conv2D` layers
  - `BatchNormalization` for training stability
  - `MaxPooling2D` for spatial downsampling
  - `Dropout` for regularization
- `GlobalAveragePooling2D` was used instead of `Flatten` to reduce overfitting and improve generalization
- Final classification layer with `Softmax` activation for multiclass output

This model served as a baseline and was trained and evaluated alongside transfer learning models.

---

### 📦 2. Pre-trained Transfer Learning Models

The following pre-trained models were fine-tuned on the fish dataset:

#### 🔹 VGG16
- Known for its simplicity and depth
- Fine-tuned top layers for specific fish species classification

#### 🔹 ResNet50
- Residual connections help prevent vanishing gradients in deep networks
- Very effective in capturing deeper features in images

#### 🔹 MobileNetV2
- Lightweight and optimized for mobile/embedded use
- Fast training and inference with good accuracy

#### 🔹 InceptionV3
- Uses factorized convolutions and aggressive dimensionality reduction
- Highly efficient and accurate for image classification tasks

#### 🔹 EfficientNetB0
- Balances network depth, width, and resolution
- Achieves high accuracy with fewer parameters

Each pre-trained model was used with ImageNet weights and customized by replacing the top layers to suit the number of fish classes.

---

## 📈 Model Training & Evaluation

All six models (1 custom CNN and 5 pre-trained models) were trained and evaluated using a multiclass fish image dataset. The models were assessed using the following key metrics:

- **Accuracy** – Overall percentage of correct predictions
- **Macro Precision** – Precision averaged across all classes
- **Macro Recall** – Recall averaged across all classes
- **Macro F1-Score** – Harmonic mean of macro precision and recall

Early stopping and validation monitoring were used during training to prevent overfitting. Training was accelerated using a GPU (NVIDIA GTX 1650) with CUDA and cuDNN enabled.

---

### 🧪 Evaluation Results

| Model             | Accuracy  | Macro Precision | Macro Recall | Macro F1-Score |
|------------------|-----------|------------------|---------------|----------------|
| **ResNet50**         | 0.207     | 0.2699           | 0.2595        | 0.1699         |
| **MobileNetV2**      | 0.9918    | 0.9651           | 0.9931        | 0.9764         |
| **InceptionV3**      | 0.9354    | 0.8902           | 0.9348        | 0.9006         |
| **EfficientNetB0**   | 0.1026    | 0.0093           | 0.0909        | 0.0169         |
| **VGG16**            | 0.7829    | 0.7548           | 0.8013        | 0.7325         |
| **CNN (Scratch)**    | 0.8926    | 0.8478           | 0.8813        | 0.8326         |

---

### 🏆 Best Model

✅ **MobileNetV2** outperformed all other models in terms of accuracy and overall F1-score, making it the best candidate for deployment.  
It was selected for the final Streamlit web application.

The custom CNN model also performed competitively, proving the effectiveness of the architecture designed from scratch.


## 💾 Model Saving Strategy

After training, each model was evaluated on the test dataset, and its performance metrics (accuracy, precision, recall, F1-score) were logged. The following strategy was adopted for model saving and selection:

### ✅ Individual Model Saving
Each trained model was saved using the `.h5` format in the `models/` directory using the Keras `model.save()` method:

```python
model.save(f"models/fish_{model_name}.h5")
```

## 🧪 Streamlit App Features

To make the model accessible to non-technical users, a clean and interactive web interface was built using **Streamlit**. This app enables users to classify fish species in real time by simply uploading an image.

### 🎯 Key Features

- 📤 **Image Upload**  
  Users can upload any fish image directly through the browser interface.

- 🤖 **Real-Time Prediction**  
  Once the image is uploaded, the app processes it and returns the predicted fish species with high confidence.

- 📊 **Confidence Score**  
  Displays model prediction probability to indicate how confident the model is.

- ⚠️ **Low Confidence Warning**  
  If the confidence score falls below a threshold, the app warns that the image may not belong to any known fish class.

- 💡 **Fully Dynamic Backend**  
  The app loads the **best model dynamically** and does not rely on hardcoded class labels.

- 🎯 **Lightweight & Fast**  
  Ideal for real-time predictions even on limited-resource cloud servers.

---

### 🌐 Streamlit App UI Screenshot

![image](https://github.com/user-attachments/assets/cea1b8ae-4bb7-4bda-ad48-794c2606775a)

## 🐳 Dockerization & Deployment

To ensure platform-independent and scalable deployment, the Streamlit app was containerized using **Docker** and deployed on an **AWS EC2 instance (Amazon Linux)** using SSH.

---

### 💻 Steps Followed:

#### 1️⃣ SSH into AWS EC2 Instance

Connected securely to the EC2 instance using a private key:
```bash
ssh -i "FishStreamlitApp.pem" ec2-user@<your-ec2-public-ip>
```

#### 2️⃣ Transferred Project Files

Used SCP to transfer necessary project files (model, app, Dockerfile, etc.) from local machine to EC2:
```bash
scp -i "FishStreamlitApp.pem" -r \
"C:/path/to/your/app.py" \
"C:/path/to/Dockerfile" \
"C:/path/to/models" \
"C:/path/to/requirements.txt" \
ec2-user@<your-ec2-public-ip>:~/Fish_App/
```

#### 3️⃣ Built Docker Image on EC2

Navigated into the app directory and built the image:
```bash
cd Fish_App
docker build -t fish-classifier-app .

```

#### 4️⃣ Ran Docker Container

Exposed the app on port 8501 (Streamlit’s default port):
```bash
docker run -d -p 8501:8501 fish-classifier-app .
```

Checked container status using:
```bash
docker ps
```

#### 🌐 Access the Web App

The deployed Streamlit app was accessible via:
```bash
http://<your-ec2-public-ip>:8501

```

## ✅ Summary of Achievements

This project demonstrates the end-to-end development and deployment of a real-time fish species classification system using deep learning and cloud-based tools. Here's a snapshot of the major accomplishments:

- 🧠 **Developed a Custom CNN Model**  
  Designed and trained a deep Convolutional Neural Network from scratch for baseline performance.

- 🔁 **Implemented 5 Transfer Learning Models**  
  Fine-tuned state-of-the-art pre-trained architectures like MobileNetV2, ResNet50, VGG16, InceptionV3, and EfficientNetB0.

- 📈 **Conducted In-Depth Model Evaluation**  
  Compared all models using accuracy, precision, recall, and F1-score to identify the most effective model.

- 🏆 **Automated Best Model Selection & Saving**  
  Created a dynamic evaluation system to select the top-performing model (MobileNetV2) and save it for deployment.

- 🧪 **Built a Streamlit Web Application**  
  Developed a user-friendly interface to upload fish images and view real-time classification results with confidence scores.

- 🐳 **Containerized with Docker**  
  Dockerized the entire application for seamless deployment and portability.

- ☁️ **Deployed on AWS EC2 (Amazon Linux)**  
  Successfully hosted the app on a cloud server accessible via a public IP.

- 📦 **Handled Dataset Challenges**  
  Tackled class imbalance using targeted augmentation and leveraged `ImageDataGenerator` for efficient training.

Each of these achievements contributes to a scalable, production-ready, and cloud-accessible AI solution for real-world fish image classification tasks.






