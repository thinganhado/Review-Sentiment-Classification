# Sentiment Analysis of Reviews

This project builds a **binary classifier** to predict **positive** or **negative** sentiment for short reviews from **Amazon**, **IMDB**, and **Yelp**. It combines **TF IDF features**, a **LinearSVC** classifier, **cross validation**, and **hyperparameter search**.

## Dataset
- **Source**: [Sentiment Labelled Sentences Data Set, Kaggle](https://www.kaggle.com/datasets/marklvl/sentiment-labelled-sentences-data-set)
- **About**: Short **sentences** from **Amazon**, **IMDB**, and **Yelp**, each sentence has a **binary label** [**0** negative, **1** positive]. Commonly used for **baseline sentiment classification**, **feature exploration**, and **model benchmarking**.

## Project Overview

**Goal**: Predict review sentiment as **positive** or **negative**.

**Techniques Used**:
- **Data cleaning and standardisation**.

**Feature engineering**:
- **Word TF IDF** with **unigrams** and **bigrams**.
- **Character TF IDF** with **character n grams** in the range **3 to 5**.
- **Domain one hot feature** to encode **amazon**, **imdb**, **yelp**.

**Model training and selection**:
- **LinearSVC** with **GridSearchCV**, **5 fold CV**, and an **80 to 20** train to validation split for better generalisation.
- **Hyperparameters** include **C**, **loss**, and **class_weight**, with the best found around **C = 0.1** and **squared_hinge**.

**Evaluation**: **accuracy**, **precision**, **recall**, **F1**, and **confusion matrices** on **validation** and **test** sets.

## Findings

- **Mixed or conflicting sentiment**: Misclassifications often arise when a review contains both positive and negative cues, for example, praise followed by a negative outcome.
- **Negation sensitivity**: Accuracy **drops from about 86.2 percent to about 83.7 percent** when negation is present, **false negatives increase from about 9.1 percent to about 16.3 percent**, and **false positives drop to 0 percent** in negated sentences, indicating a conservative bias toward negative predictions under negation.
- **Review length effect**: **Longer reviews** correlate with more errors, **false negatives** tend to have the longest sentences, and the spread of lengths is wider for incorrect predictions.
- **Domain differences**: Performance is stronger on **Amazon** and **Yelp** at about **0.87 accuracy**, and lower on **IMDB** at about **0.83 accuracy**.
- **Precision versus recall trade off**: The model shows **higher precision** and **slightly lower recall** on the test set, with confusion matrices indicating more missed positives relative to false positives.
- **Potential improvements**: Add **trigrams** or **dependency features** for complex negation, **chunk long reviews** and aggregate predictions, include a **length feature**, apply **domain specific preprocessing**, or train **separate domain models**.

## Files
- `sentiment_analysis_model.ipynb`: Code and output notebook
- `sentiment_analysis_report.pdf`: Report on EDA findings and model performance
- `data/`: `x_train.csv`, `y_train.csv`, `x_test.csv` 
