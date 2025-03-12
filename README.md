# **Sentiment Analysis of Reviews**
[![Python](https://img.shields.io/badge/Python-3.x-blue.svg)](https://www.python.org/)  
[![Jupyter Notebook](https://img.shields.io/badge/Jupyter-Notebook-orange.svg)](https://jupyter.org/)  
[![Sentiment Analysis](https://img.shields.io/badge/Sentiment-Analysis-green.svg)](https://github.com/shreyasa31/Sentiment_Analysis_of_Reviews)

## **Project Overview**  
This project performs **Sentiment Analysis** on customer reviews using **Natural Language Processing (NLP)** techniques. The goal is to classify reviews into **positive, negative, or neutral sentiments** using machine learning and deep learning models.

---

## **Dataset**
The dataset consists of customer reviews sourced from **[mention source like Kaggle, Amazon, Yelp, or a custom dataset]**. It includes:
- **Text Reviews**: Customer opinions.
- **Ratings**: (If available) Star ratings for products/services.
- **Labels**: Sentiment categories - Positive, Negative, or Neutral.

The dataset is preprocessed, tokenized, and vectorized for model training.

---

## **Project Workflow**
### 1️⃣ **Data Preprocessing**
- Removed special characters, stopwords, and punctuation.
- Tokenized and lemmatized the text.
- Balanced the dataset to prevent class bias.

### 2️⃣ **Exploratory Data Analysis (EDA)**
- **Word Cloud** of frequently occurring words.
- **Sentiment distribution** across the dataset.

### 3️⃣ **Feature Engineering**
- **TF-IDF Vectorization** for machine learning models.
- **Word Embeddings (Word2Vec, GloVe)** for deep learning models.

### 4️⃣ **Model Training & Evaluation**
The following models were trained:

| **Model** | **Test Accuracy (%)** |
|-----------|----------------------|
| MLP (Multi-Layer Perceptron) | **54%** |
| CNN (Convolutional Neural Network) | **57%** |
| LSTM RNN (Long Short-Term Memory) | **51%** |

📌 **CNN performed the best with 57% accuracy.** However, the accuracy is limited due to assignment constraints.

---

## **Why is Accuracy Low?**
1. **Dataset Size & Quality**: Small datasets or noisy data can affect model learning.
2. **Feature Representation**: TF-IDF may not capture deep semantic meanings.
3. **Hyperparameter Constraints**: Limited tuning of layers, epochs, and optimizers due to assignment requirements.
4. **Model Choice**: LSTM underperforms on smaller datasets if sequence dependencies are weak.

---

## **How to Improve Accuracy?**
✅ **Enhance Data Preprocessing**:
- Remove **rare words & outliers**.
- Use **advanced NLP techniques like Named Entity Recognition (NER)**.

✅ **Feature Engineering**:
- Replace TF-IDF with **Word2Vec, GloVe, or BERT embeddings** for richer feature extraction.

✅ **Better Model Selection**:
- **Use CNN + LSTM hybrid** to capture both local & sequential text patterns.
- **Fine-tune hyperparameters**: Learning rate, batch size, dropout, and optimizer (Adam, RMSprop).

✅ **Preferred Model**:
- If constrained, **CNN is preferred** as it captures local patterns in text better than MLP.
- If more flexibility is allowed, **BERT-based models (e.g., DistilBERT, RoBERTa) will significantly improve results**.

---

## **Installation & Setup**
### **1️⃣ Clone the Repository**
```bash
git clone https://github.com/shreyasa31/Sentiment_Analysis_of_Reviews.git
cd Sentiment_Analysis_of_Reviews
```

### **2️⃣ Install Dependencies**
```bash
pip install -r requirements.txt
```
Ensure you have **Jupyter Notebook** installed:
```bash
pip install notebook
```

### **3️⃣ Run the Jupyter Notebook**
```bash
jupyter notebook
```
Open **Kamath_ShreyaSadashiva_FinalProject.ipynb** and run all cells.

---

## **Results & Insights**
- **CNN (57%)** outperformed MLP (54%) and LSTM RNN (51%).
- **Common words in positive reviews**: ‘great’, ‘love’, ‘amazing’.
- **Common words in negative reviews**: ‘bad’, ‘worst’, ‘disappointed’.
- BERT-based models **(not included in assignment)** would likely improve accuracy beyond 80%.

---

## **Technologies Used**
- **Python** 🐍
- **NLTK & SpaCy** (Text preprocessing)
- **Scikit-learn** (Machine Learning models)
- **TensorFlow/PyTorch** (Deep Learning models)
- **Matplotlib & Seaborn** (Data Visualization)

---

## **Future Improvements**
🚀 **Fine-tune models with hyperparameter optimization**.  
🚀 **Use transformers (BERT, DistilBERT) for contextual embeddings**.  
🚀 **Deploy as a web app using Flask/FastAPI for real-time sentiment analysis**.  

---
