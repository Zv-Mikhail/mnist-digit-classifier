# 🧠 MNIST Digit Classifier with Drawing App 🖌️

This project uses a fully connected neural network trained on the MNIST dataset to recognize hand-drawn digits.  
You can draw a digit with your mouse and get a prediction in real time!

---

## 🔧 Project Contents

| File            | Description                            |
|-----------------|----------------------------------------|
| `notebook.ipynb` | Jupyter notebook used to train the model |
| `prog.py`        | Drawing app for predicting your digits  |
| `my_model.pth`   | Pretrained PyTorch model weights        |
| `requirements.txt` | List of Python dependencies           |

---

## 🖥️ How to Run

1. Install dependencies:

pip install -r requirements.txt

2. Run the drawing app:

python prog.py

## ✏️ Drawing Instructions

* Draw a digit (0–9) by clicking and dragging your mouse

* Press Spacebar to clear the canvas

## 📊 Model Performance
* Achieved ~98% test accuracy on MNIST
* Fully connected neural network with 4 hidden layers
* Trained using Adam optimizer and CrossEntropyLoss

## 🐍 Python Version

3.11.13

## 📷 Example Prediction

![alt text](image-1.png)