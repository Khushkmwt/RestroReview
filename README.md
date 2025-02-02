# 🌟 Sentiment Analysis of Restaurant Reviews

## 📌 Project Overview
This project implements **Natural Language Processing (NLP)** with a **Naive Bayes classifier** to analyze restaurant reviews and classify them as **positive** or **negative**.

---

## 📂 Dataset
- **File:** `Restaurant_Reviews.tsv`
- **Size:** 1000 reviews
- **Format:** Tab-separated values (TSV)
- **Labels:** 1 (Positive) / 0 (Negative)

---

## 🛠️ Installation
To install the required dependencies, run:
```bash
pip install numpy pandas nltk scikit-learn matplotlib
```

---

## 🔄 Workflow

### 📥 1. Import Required Libraries
Essential libraries for **data handling**, **NLP**, and **machine learning** are imported.

### 📊 2. Load and Explore the Dataset
We use Pandas to load the dataset:
```python
dataset = pd.read_csv('Restaurant_Reviews.tsv', delimiter='\t', quoting=3)
```

### 📝 3. Text Preprocessing
- **Removing special characters**
- **Lowercasing text**
- **Tokenization** (splitting into words)
- **Removing stopwords** (keeping 'not' for sentiment)
- **Applying stemming** (reducing words to root forms)

### 🔢 4. Convert Text to Features
We use **`CountVectorizer`** to transform text into numerical vectors:
```python
cv = CountVectorizer(max_features=1500)
X = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:, -1].values
```

### 🤖 5. Train the Model
We split the dataset and train a **Naive Bayes classifier**:
```python
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
```

### 📊 6. Model Evaluation
Performance is measured using **confusion matrix** and **accuracy score**:
```python
from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
accuracy_score(y_test, y_pred)
```

### 🧐 7. Predicting a New Review
To classify a new review:
```python
new_review = 'I love this restaurant so much'
new_review = re.sub('[^a-zA-Z]', ' ', new_review)
new_review = new_review.lower().split()
ps = PorterStemmer()
all_stopwords = stopwords.words('english')
all_stopwords.remove('not')
new_review = [ps.stem(word) for word in new_review if word not in set(all_stopwords)]
new_review = ' '.join(new_review)
new_corpus = [new_review]
new_X_test = cv.transform(new_corpus).toarray()
new_y_pred = classifier.predict(new_X_test)
print(new_y_pred)
```

---

## 🎯 Results
- **Model Accuracy:** **73%**
- Example Prediction: The review **"I love this restaurant so much"** was classified as **negative**.

---

## ▶️ How to Run the Project
1. Place `Restaurant_Reviews.tsv` in the project directory.
2. Run the script:
   ```bash
   python sentiment_analysis.py
   ```
3. View **accuracy and predictions** in the console.

---

## ✨ Author
**Dilkhush Kumawat** 🚀

---

This README is crafted to be **clean, visually appealing, and easy to understand**. Hope you find it helpful! 🎉

