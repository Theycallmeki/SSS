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

## Task 1 - Load the dataset

I loaded the dataset using `pd.read_csv`, and it displayed tweets with their sentiment labels.

| target | text |
|--------|------|
|0 | @switchfoot http://twitpic.com/2y1zl - Awww, t...|
|0 | is upset that he can't update his Facebook by ...|
|0 | @Kenichan I dived many times for the ball. Man...|
|0 |  my whole body feels itchy and like its on fire |
|0 | @nationwideclass no, it's not behaving at all....|
|0 |                      @Kwesidei not the whole crew |
|0 |                                        Need a hug|
|0 | @LOLTrish hey  long time no see! Yes.. Rains a...|
|0 |              @Tatiana_K nope they didn't have it |
|0 |                         @twittera que me muera ? |

---

## Task 2 - Keep only the positive and negative tweets

I filtered the dataset to remove neutral tweets and kept only positive and negative. The counts are:

- **Positive tweets (1): 51,150 (50.56%)**  
- **Negative tweets (0): 50,009 (49.44%)**

---

## Task 3 - Preprocess the text

I preprocessed the dataset by converting all text to lowercase.

|index | target | text|
|------|--------|-------------------------------------------------|
|0     |  0     |@switchfoot http://twitpic.com/2y1zl - awww, t...|
|1     |  0     |is upset that he can't update his facebook by ...|
|2     |  0     |@kenichan i dived many times for the ball. man...|
|3     |  0     |   my whole body feels itchy and like its on fire|
|4     |  0     |@nationwideclass no, it's not behaving at all....|

---

## Task 4 - Train-test split 

I split the dataset into training and testing sets (80% train, 20% test) to prepare for model evaluation.

- **Training set:** 80,927 samples  
- **Test set:** 20,232 samples  

---

## Task 5 - Perform TF-IDF vectorization

I applied TF-IDF vectorization with a 5,000 feature limit, fitting on the training set and transforming both train and test sets.

- **Training shape:** (80,927, 5000)  
- **Test shape:** (20,232, 5000)  

---

## Task 6 - Train 3 models: BernoulliNB, LinearSVC, LogisticRegression

The three models were trained and evaluated on the test set.  

**Model Accuracies (Test Set):**  
- BernoulliNB: 76.35%  
- LinearSVC: 78.29%  
- LogisticRegression: 78.60%  

---

## Task 7 - Inference

I tested the models with three sample tweets.  

**1. Text:** *"I love you!"*  
- BNB → Positive  
- LSVC → Positive  
- LR → Positive  

**2. Text:** *"I hate you but I love you also."*  
- BNB → Positive  
- LSVC → Positive  
- LR → Positive  

**3. Text:** *"I love your code, it's so clean. :)"*  
- BNB → Positive  
- LSVC → Positive  
- LR → Positive  

---

## Observation  

- All models consistently labeled the sample tweets as **Positive**, showing agreement.  
- Logistic Regression achieved the best accuracy (78.60%), followed closely by LinearSVC (78.29%), and then BernoulliNB (76.35%).  
- Logistic Regression and LinearSVC are better at capturing subtle positive tone, while BernoulliNB tends to misclassify positives more often.  
