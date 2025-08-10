

# 📝 Twitter Sentiment Analysis with Deep RNN, LSTM, and GRU

## 📌 Project Overview

This project performs **sentiment analysis** on Twitter data using deep learning models built with **Keras/TensorFlow**.
We preprocess the tweets (lowercasing, removing punctuation, stopwords, emojis, and lemmatization) and train **three deep sequence models**:

* 🔹 **Deep RNN** (SimpleRNN layers)
* 🔹 **Deep LSTM** (Long Short-Term Memory)
* 🔹 **Deep GRU** (Gated Recurrent Unit)

The aim is to compare the performance of these architectures for **text classification**.

---

## 📂 Dataset

* **Training & Validation Data:** Twitter posts with labeled sentiments.
* **Columns Used:** `Text` (tweet content) and `Label` (sentiment).
* **Unused Columns:** `ID`, `Entity` (dropped during preprocessing).

---

## ⚙️ Preprocessing Steps

1. Convert text to **lowercase**.
2. Remove **HTML tags**, **URLs**, **punctuation**, **digits**, and **extra spaces**.
3. Remove **stopwords** and **emojis**.
4. Perform **lemmatization** to reduce words to their base form.
5. **Tokenization & Padding** for neural network input.

---

## 🏗 Model Architectures

### 1️⃣ Deep RNN

```python
Embedding(vocab_size, 128)
SimpleRNN(128, return_sequences=True)
Dropout(0.3)
SimpleRNN(64)
Dropout(0.3)
Dense(1, activation='sigmoid')
```

### 2️⃣ Deep LSTM

```python
Embedding(vocab_size, 128)
LSTM(128, return_sequences=True)
Dropout(0.3)
LSTM(64)
Dropout(0.3)
Dense(1, activation='sigmoid')
```

### 3️⃣ Deep GRU

```python
Embedding(vocab_size, 128)
GRU(128, return_sequences=True)
Dropout(0.3)
GRU(64)
Dropout(0.3)
Dense(1, activation='sigmoid')
```

---

## 📊 Training

* **Loss:** `binary_crossentropy`
* **Optimizer:** `adam`
* **Metrics:** `accuracy`
* **Batch Size:** 64
* **Epochs:** 5 (with Early Stopping to prevent overfitting)

---

## 📈 Evaluation & Prediction

* Evaluate model performance on test data.
* Predict sentiment for new/unseen tweets (`val_df_test`).

Example prediction flow:

```python
val_seq = tokenizer.texts_to_sequences(val_df_test["Text"])
val_pad = pad_sequences(val_seq, maxlen=max_len)
predictions = (model.predict(val_pad) > 0.5).astype(int)
```

---

## 🛠 Requirements

```
pandas
numpy
nltk
emoji
tensorflow
scikit-learn
```

---

## 🚀 How to Run

1. Clone the repository:

```bash
git clone https://github.com/yourusername/twitter-sentiment-analysis.git
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Run the notebook:

```bash
jupyter notebook twitter-sentiment-analysis.ipynb
```

---

## 📌 Future Improvements

* Add **BERT** or other Transformer-based models.
* Use **pre-trained word embeddings** like GloVe or Word2Vec.
* Hyperparameter tuning for better accuracy.

---

Do you want me to also create a **diagram** showing the complete pipeline — preprocessing → tokenization → model training → prediction — for the README? That would make it more visual and appealing.

