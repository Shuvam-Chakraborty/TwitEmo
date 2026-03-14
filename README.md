# 🐦 TwitEmo — Twitter Sentiment Analysis

> Classifying the emotional pulse of Twitter using NLP, TF-IDF, and XGBoost  
> **71% Test Accuracy · ROC-AUC 0.849 · 27,000+ Tweets**

---

## 📌 Overview

**TwitEmo** is an end-to-end NLP pipeline that classifies tweets into **positive**, **negative**, or **neutral** sentiment. Built as part of an AI & ML Lab project, it applies classical machine learning — TF-IDF feature extraction with LinearSVC and XGBoost classifiers — on a real-world Twitter dataset.

The final XGBoost model achieves **71.05% test accuracy** and a **weighted ROC-AUC of 0.849**, outperforming benchmark scripts on the same dataset while remaining fully interpretable and lightweight.

---

## 📊 Results at a Glance

| Model | CV Accuracy | Test Accuracy | Weighted ROC-AUC | Macro F1 |
|---|---|---|---|---|
| LinearSVC + TF-IDF | 69.08% | 69.45% | 0.8405 | 0.70 |
| **XGBoost + TF-IDF** | **69.43%** | **71.05%** | **0.8492** | **0.71** |

> ✅ XGBoost selected as final model — better generalization, no overfitting, fully interpretable.

---

## 🗂️ Project Structure

```
TwitEmo/
│
├── twitemo.ipynb          # Main notebook (all 6 steps)
└── README.md
```

> **Note:** The trained models (`xgb_model.pkl`, `svm_model.pkl`, `tfidf_vectorizer.pkl`) are not included. Run the notebook on Kaggle to train and save them locally.

---

## 🔄 Pipeline

```
Raw Tweets (27,481)
    │
    ▼
Step 1 — Dataset Exploration
    │   4 columns: textID, text, selected_text, sentiment
    │   3 classes: neutral (40.5%) · positive (31.3%) · negative (28.4%)
    │   Avg tweet length: 68 chars · Max: 141 chars
    ▼
Step 2 — Text Preprocessing
    │   Lowercasing · URL & mention removal
    │   Lemmatization (WordNetLemmatizer)
    │   Stopword removal — negations preserved: not, no, don't...
    │   56 empty strings dropped after cleaning
    ▼
Step 3 — Train / Val / Test Split (Stratified)
    │   80% Train (21,939) · 10% Val (2,742) · 10% Test (2,743)
    │   Class balance preserved across all splits
    ▼
Step 4 — TF-IDF Feature Extraction
    │   Train + Val merged → 24,681 samples for model training
    │   Unigrams + Bigrams + Trigrams · 24,327 features extracted
    │   sublinear_tf=True · min_df=2 · max_df=0.95
    ▼
Step 5 — Model Training (5-Fold Stratified CV)
    │   Model 1: LinearSVC (CalibratedClassifierCV, C=0.8)
    │   Model 2: XGBoost (500 trees, hist method, balanced weights)
    ▼
Step 6 — Evaluation
        Classification Report · Confusion Matrix
        ROC-AUC · Per-class F1 · Custom Tweet Predictions
```

---

## 🧪 Per-Class Performance (XGBoost)

| Sentiment | Precision | Recall | F1-Score | Support |
|---|---|---|---|---|
| Negative | 0.74 | 0.63 | 0.68 | 778 |
| Neutral | 0.65 | 0.75 | 0.69 | 1107 |
| Positive | 0.79 | 0.73 | 0.76 | 858 |
| **Weighted Avg** | **0.72** | **0.71** | **0.71** | **2743** |

### Key Observations
- **Positive** is the easiest class — strong, unambiguous signal words (*love, amazing, great*) push F1 to 0.76
- **Neutral** has the highest recall (0.75) but lowest precision (0.65) — the model over-predicts neutral for ambiguous tweets
- **Negative** has the lowest recall (0.63) — sarcasm and subtle negativity (*"not bad, could be better"*) are hard to catch with TF-IDF alone

---

## 🔍 Custom Tweet Predictions

```
Tweet                                          Predicted    Confidence
----------------------------------------------------------------------
I absolutely love this!                         positive        85.7%
This is the worst experience ever               negative        64.1%
The movie was just okay, nothing special         neutral        47.0%
Amazing performance by the team today!!         positive        72.2%
I'm really disappointed with the service        negative        51.9%
Not bad, could be better though                 negative        44.1%
so happy right now!!!                           positive        86.0%
ugh this is so frustrating                      negative        69.0%
```

> Low confidence on ambiguous tweets (*47%, 44.1%*) shows the model is **well-calibrated** — it knows when it is uncertain.

---

## 🛠️ Tech Stack

| Category | Tools |
|---|---|
| Language | Python 3.12 |
| Data | pandas, numpy |
| NLP Preprocessing | NLTK (stopwords, WordNetLemmatizer) |
| Feature Extraction | scikit-learn TfidfVectorizer (ngrams 1–3) |
| Models | scikit-learn LinearSVC, XGBoost |
| Evaluation | scikit-learn metrics, matplotlib, seaborn |
| Environment | Kaggle Notebooks (CPU) |

---

## 📁 Dataset

**Twitter Tweets Sentiment Dataset** — [Kaggle](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

- 27,481 tweets labeled as `positive`, `negative`, or `neutral`
- 4 columns: `textID`, `text`, `selected_text`, `sentiment`
- 1 null row dropped → **27,480 usable samples**
- Column used for modeling: `text` (full tweet)

```
sentiment
neutral     11,118  (40.5%)
positive     8,582  (31.3%)
negative     7,781  (28.4%)
```

---

## 🧠 Key Design Decisions

**Why TF-IDF over word embeddings?**  
With ~22k training samples, learned embeddings risk overfitting. TF-IDF with trigrams captures multi-word sentiment phrases (*"not very good", "really love it"*) without needing large data.

**Why XGBoost over SVM?**  
Both scored similarly in CV (~69%), but XGBoost generalized better on the unseen test set (71.05% vs 69.45%) and provides feature importance for interpretability.

**Why negations are preserved in stopword removal?**  
Words like *not, no, don't, couldn't* completely flip sentiment meaning. Removing them causes *"not good"* to be treated identically to *"good"*.

**Why train + val merged for model training?**  
SVM and XGBoost use 5-fold cross-validation internally, so the validation set is folded back into training to maximize data usage. The held-out test set is only touched at final evaluation.

---

## 📈 What Could Be Improved

- **Transformer fine-tuning** — BERTweet (pretrained on 850M tweets) would likely push accuracy to 78–82%
- **Sarcasm detection** — Current model struggles with implicit negativity
- **More negative samples** — Slight class imbalance hurts negative recall (0.63)
- **Web app deployment** — Flask/Streamlit interface for real-time tweet classification

---

## 🚀 How to Run

> ⚠️ **This notebook is designed to run on Kaggle.** No GPU needed — TF-IDF + XGBoost runs entirely on CPU.

**Step 1 — Get the dataset**  
Go to the dataset page and click **Download**:  
👉 [https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset](https://www.kaggle.com/datasets/yasserh/twitter-tweets-sentiment-dataset)

**Step 2 — Create a new Kaggle Notebook**  
Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook** → Upload `twitemo.ipynb`

**Step 3 — Add the dataset to your notebook**  
In the Kaggle notebook sidebar → **Add Data** → search for:  
`Twitter Tweets Sentiment Dataset` by yasserh → Add it

**Step 4 — Run All**  
Click **Run All** or go cell by cell with `Shift + Enter`

The dataset will be available at:
```
/kaggle/input/datasets/yasserh/twitter-tweets-sentiment-dataset/Tweets.csv
```

---

## 📝 License

This project is for academic purposes as part of an AI & ML Lab course.

---

<p align="center">Built with 🐦 tweets, 📊 trigrams, and 🌳 gradient boosting</p>
