# 📚 Next Word Prediction using LSTM & Bidirectional RNN

This project implements a **deep learning model** for predicting the next word in a sequence of text using **LSTM (Long Short-Term Memory)** networks with **Bidirectional layers**. It is trained on the *Sherlock Holmes* text dataset to learn the structure, semantics, and flow of English sentences.

---

## 🚀 Features
- **Text Preprocessing**: Cleaning, tokenizing, and creating word sequences from raw text.  
- **Vocabulary Building**: Uses Keras `Tokenizer` with Out-Of-Vocabulary (`<OOV>`) token support.  
- **Sequence Generation**: Creates fixed-length input sequences for training.  
- **Deep Learning Model**:
  - Embedding Layer for word vector representation.
  - Two stacked Bidirectional LSTM layers for context learning.
  - Dropout for regularization.
  - Dense layers for prediction.
- **Temperature Sampling**: Controls randomness in generated text.  
- **Training Optimization**: Early stopping and learning rate reduction callbacks.

---

## 🛠 Tech Stack
- **Python**
- **TensorFlow / Keras**
- **NumPy**
- **Pandas**
- **NLTK**
- **Matplotlib**
- **Kaggle API** (for dataset download)

---

## 📂 Dataset
The project uses the [Next Word Prediction Dataset](https://www.kaggle.com/datasets/ronikdedhia/next-word-prediction) from Kaggle, containing `1661-0.txt` (*Sherlock Holmes* novel).  
Downloaded via Kaggle API in Google Colab.

---

## 📊 Model Summary
- **Embedding Dimension**: 128  
- **LSTM Layers**: 256 + 128 units (Bidirectional)  
- **Dropout**: 0.3  
- **Dense**: 128 units + Softmax output  
- **Optimizer**: Adam (learning rate: 0.001)  
- **Loss Function**: Categorical Crossentropy  

---

## 📈 Training
- **Batch Size**: 256  
- **Epochs**: Up to 50 (with EarlyStopping)  
- **Validation Split**: 10%

---

## 🔮 Example Prediction
**Seed Text:**  

