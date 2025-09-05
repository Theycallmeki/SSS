# 📊 Twitter Sentiment Analysis

**Name:** Laxamana, KIAN JACOB LAXAMANA  
**Dataset Used:** [TweetFeels 100k](https://huggingface.co/datasets/mnemoraorg/tweetfeels-100k)  
**Tools:** Python 3.8+, pandas, scikit-learn, scipy, joblib  

---

## Overview

This project implements an end-to-end sentiment analysis pipeline on Twitter data.  
The objective is to classify tweets as either **positive** or **negative** using machine learning models trained on the **TweetFeels 100k** dataset.  

The process is divided into seven tasks:  
1. Load the dataset  
2. Keep only positive/negative tweets  
3. Preprocess the text  
4. Train-test split  
5. TF-IDF vectorization  
6. Train and save models  
7. Inference with custom tweets  

---

## Task 1: Load the Dataset

The dataset was loaded using `pandas.read_csv()` from a local CSV file (`tweetfeels.csv`) downloaded from Hugging Face.  
The load completed without errors, confirming that the file path, delimiter, and encoding were handled correctly.  

From the available schemas, only the sentiment label and tweet text were retained to standardize downstream processing.

---

## Task 2: Keep Only Positive and Negative Tweets

To focus strictly on binary sentiment classification, the dataset was filtered so that only tweets labeled as positive (`4`) or negative (`0`) were retained, while neutral tweets (`2`) were dropped entirely.  

After filtering, the labels were remapped into a consistent binary scheme where `0` represented negative sentiment and `1` represented positive sentiment.  

This mapping was necessary to simplify the classification task and ensure uniformity across the pipeline.  

The decision to remove neutral tweets was made because they often introduce ambiguity, as their interpretation can vary depending on annotator guidelines, and this ambiguity can negatively affect model performance.  

After the filtering and remapping process, the class distributions were recomputed to confirm that both positive and negative classes remained well represented. 

---

## Task 3: Preprocess the Text

Tweets were normalized by converting all text to lowercase to reduce vocabulary sparsity and improve feature sharing across similar tokens.

**Approach (minimal, reproducible):**
```python
def _lowercase(text: str) -> str:
    return text.lower()

---

Task 4: Train-Test Split
The dataset was split into 80% training and 20% testing using scikit-learn’s train_test_split().
Stratified sampling was applied to maintain class balance.

Set	Proportion	Size (Approx.)
Training	80%	~80,000 tweets
Testing	20%	~20,000 tweets

Task 5: TF-IDF Vectorization
Tweets were converted into numerical features using TfidfVectorizer.

Configuration:

max_features = 5000

ngram_range = (1, 2) → includes unigrams and bigrams

The vectorizer was fit on the training data and then applied to both training and test sets.

Task 6: Train and Save 3 Models
Three machine learning models were trained on the TF-IDF vectors:

Bernoulli Naive Bayes

Linear Support Vector Classifier (LinearSVC)

Logistic Regression

Each model was trained, evaluated, and saved using joblib.

Model	Accuracy (%)	Saved File
Bernoulli Naive Bayes	76.35%	bnb.pkl
Linear SVC	78.29%	lsvc.pkl
Logistic Regression	78.60%	lr.pkl

Observation: Logistic Regression achieved the highest accuracy.

Task 7: Inference
Three custom-written tweets were tested on all three models.

#	Tweet	BernoulliNB	LinearSVC	LogisticRegression
1	I love you!	Positive	Positive	Positive
2	I hate you but I love you also.	Positive	Positive	Positive
3	I love your code, it's so clean. :)	Positive	Positive	Positive

Observations:

All models consistently labeled the sample tweets as Positive, showing agreement in inference.

Logistic Regression and LinearSVC produced the best overall test accuracy in evaluation, with Logistic Regression slightly leading.

Bernoulli Naive Bayes underperformed compared to the other two models, but still achieved a respectable baseline performance above 76%.
