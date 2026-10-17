# Summative-Lab--Analytics-Firm-Clients
# Customer Reviews Analysis: NLP, Time Series & Neural Networks

An end-to-end analysis of Amazon customer reviews (mostly Nook and Kindle e-readers/tablets) for an analytics firm client. The notebook `C09_M08.ipynb` covers three parts:

1. **NLP**: text cleaning, common words and bigrams, and visualizations
2. **Time Series**: monthly review volume and ratings, with a 3-month rolling average
3. **Neural Network**: predicting 1–5 star ratings from review text

## Key Findings

- **Sentiment is mostly positive.** About 58% of reviews are 5-star, and phrases like "work great" and "work well" are common. Recurring concerns are battery life, touch screen, SD card, and customer service.
- **Review activity grew over time.** Volume rose from 131 reviews in 2010 to 325 in 2013, peaking at 46 in December 2012 (likely holiday purchases). Average rating climbed from about 3.9 to about 4.3.
- **The model is limited by class imbalance.** The neural network reaches **54% accuracy**. It predicts 5-star reviews best (F1 0.70) but struggles with 2-, 3-, and 4-star reviews, often defaulting to 5 stars.

## Dataset

`amazon_reviews_lab.csv` with (at least) these columns:

| Column | Description |
|---|---|
| `review_text` | Full text of the review |
| `rating` | Star rating (1–5) |
| `timestamp` | Date of the review |

> The CSV is not included in this repo. Place it in the same folder as the notebook.

## Methods

- **NLP:** lowercasing, URL and punctuation removal, NLTK tokenization, stopword removal, lemmatization, then `Counter` for word and bigram frequencies.
- **Time series:** `pandas` monthly resampling (count and mean rating) and a 3-month rolling mean.
- **Neural network:** TF-IDF (5,000 features) → Dense(256) → Dropout(0.3) → Dense(128) → Dropout(0.3) → Dense(5, softmax). Trained with Adam and sparse categorical cross-entropy for 10 epochs, using an 80/20 stratified split.

## Setup

```bash
pip install pandas numpy matplotlib seaborn nltk scikit-learn tensorflow jupyter
jupyter notebook C09_M08.ipynb
```

NLTK resources (`punkt`, `stopwords`, `wordnet`) download automatically on the first run.

## Possible Improvements

- Address class imbalance (class weights, resampling) or collapse ratings into negative / neutral / positive.
- Try word embeddings or a pretrained transformer instead of TF-IDF.
- Add sentiment scoring and analyze themes by product (Nook vs. Kindle).
