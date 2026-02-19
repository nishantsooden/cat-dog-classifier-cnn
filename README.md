# 🐶🐱 Cat vs Dog Classifier using CNN and Streamlit

An end-to-end Deep Learning project that classifies images of Cats and Dogs using a Convolutional Neural Network (CNN) built with TensorFlow/Keras and deployed using Streamlit.

This project demonstrates the complete machine learning workflow including data preprocessing, model training, evaluation, saving, and deployment as an interactive web application.

---

## 🚀 Features

- CNN-based image classification
- TensorFlow/Keras model training
- GPU acceleration support (RTX 4060 compatible)
- Real-time image prediction
- Streamlit web app deployment
- Clean and modular project structure

---

## 🧠 Model Details

- Framework: TensorFlow / Keras
- Architecture: Convolutional Neural Network (CNN)
- Input Size: 224 × 224 × 3
- Output: Binary classification (Cat or Dog)
- Training Accuracy: ~90%
- Validation Accuracy: ~80%+

---

## 📂 Project Structure
CNN/
│
├── Dataset/
│ ├── Train/
│ └── Test/
│
├── Model/
│ └── cat_dog_model.keras
│
├── Src/
│ └── app.py
│
├── Notebooks/
│ └── CNN.ipynb
│
├── requirements.txt
└── README.md


---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/nishantsooden/cat-dog-classifier-cnn.git
cd cat-dog-classifier-cnn-streamlit


Create environment (recommended):

conda create -n cnn_env python=3.10
conda activate cnn_env

Install dependencies:

pip install -r requirements.txt
▶️ Run the Streamlit App
streamlit run Src/app.py