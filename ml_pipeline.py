#!/usr/bin/env python3
"""
Support Ticket Classification - ML Pipeline
============================================
Standalone script for training and evaluating the ticket classification models.
Part of Future Interns ML Task 2.

Usage:
    python ml_pipeline.py --train    # Train all models
    python ml_pipeline.py --predict  # Interactive prediction mode
    python ml_pipeline.py --evaluate # Evaluate model performance
"""

import argparse
import re
import string
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import joblib
import warnings
warnings.filterwarnings('ignore')

# Text preprocessing function
def preprocess_text(text):
    """Clean and preprocess text data for ML models."""
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text)
    # Remove email addresses
    text = re.sub(r'\S+@\S+', '', text)
    # Remove ticket IDs and account IDs
    text = re.sub(r'tkt-\d+|acc-\d+|inv-\d+', '', text)
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    # Simple tokenization
    tokens = text.split()
    # Remove stopwords
    stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 
                  'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                  'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
                  'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
                  'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
                  'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
                  'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
                  'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
                  'same', 'so', 'than', 'too', 'very', 'just', 'and', 'but', 'if', 'or',
                  'because', 'until', 'while', 'this', 'that', 'these', 'those', 'i', 'me',
                  'my', 'myself', 'we', 'our', 'ours', 'ourselves', 'you', 'your', 'yours',
                  'yourself', 'yourselves', 'he', 'him', 'his', 'himself', 'she', 'her',
                  'hers', 'herself', 'it', 'its', 'itself', 'they', 'them', 'their',
                  'theirs', 'themselves', 'what', 'which', 'who', 'whom', 'am', 'been'}
    tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
    return ' '.join(tokens)


def load_data(filepath='data/support_tickets.csv'):
    """Load and prepare the dataset."""
    print(f"Loading data from {filepath}...")
    df = pd.read_csv(filepath)
    print(f"Loaded {len(df)} tickets")

    # Combine subject and body for text features
    df['text'] = df['subject'] + ' ' + df['body']
    df['processed_text'] = df['text'].apply(preprocess_text)

    return df


def train_models(df):
    """Train all three classification models."""
    X = df['processed_text']

    models = {}

    # 1. Category Classification
    print("\n" + "="*60)
    print("Training Category Classification Model...")
    print("="*60)

    y_cat = df['category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_cat, test_size=0.2, random_state=42, stratify=y_cat
    )

    cat_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=5000, ngram_range=(1, 2), min_df=2)),
        ('classifier', LinearSVC(C=1.0, max_iter=2000, random_state=42))
    ])

    cat_pipeline.fit(X_train, y_train)
    y_pred = cat_pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    models['category'] = cat_pipeline
    joblib.dump(cat_pipeline, 'models/category_model.pkl')
    print("Category model saved!")

    # 2. Priority Classification
    print("\n" + "="*60)
    print("Training Priority Classification Model...")
    print("="*60)

    y_pri = df['priority']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_pri, test_size=0.2, random_state=42, stratify=y_pri
    )

    pri_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    pri_pipeline.fit(X_train, y_train)
    y_pred = pri_pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    models['priority'] = pri_pipeline
    joblib.dump(pri_pipeline, 'models/priority_model.pkl')
    print("Priority model saved!")

    # 3. Sentiment Analysis
    print("\n" + "="*60)
    print("Training Sentiment Analysis Model...")
    print("="*60)

    y_sent = df['sentiment']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_sent, test_size=0.2, random_state=42, stratify=y_sent
    )

    sent_pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=3000, ngram_range=(1, 2), min_df=2)),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
    ])

    sent_pipeline.fit(X_train, y_train)
    y_pred = sent_pipeline.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

    models['sentiment'] = sent_pipeline
    joblib.dump(sent_pipeline, 'models/sentiment_model.pkl')
    print("Sentiment model saved!")

    return models


def predict_ticket(text, models):
    """Predict category, priority, and sentiment for a ticket."""
    processed = preprocess_text(text)

    category = models['category'].predict([processed])[0]
    priority = models['priority'].predict([processed])[0]
    sentiment = models['sentiment'].predict([processed])[0]

    return {
        'category': category,
        'priority': priority,
        'sentiment': sentiment
    }


def interactive_predict(models):
    """Interactive mode for testing predictions."""
    print("\n" + "="*60)
    print("Interactive Prediction Mode")
    print("Type 'quit' to exit")
    print("="*60)

    while True:
        print("\nEnter ticket text (subject + description):")
        text = input("> ").strip()

        if text.lower() == 'quit':
            break

        if not text:
            continue

        result = predict_ticket(text, models)

        print("\n" + "-"*40)
        print("PREDICTION RESULTS:")
        print("-"*40)
        print(f"Category:  {result['category']}")
        print(f"Priority:  {result['priority']}")
        print(f"Sentiment: {result['sentiment']}")
        print("-"*40)


def main():
    parser = argparse.ArgumentParser(description='Support Ticket Classification ML Pipeline')
    parser.add_argument('--train', action='store_true', help='Train all models')
    parser.add_argument('--predict', action='store_true', help='Interactive prediction mode')
    parser.add_argument('--evaluate', action='store_true', help='Evaluate model performance')
    args = parser.parse_args()

    if args.train:
        df = load_data()
        train_models(df)

    elif args.predict:
        print("Loading models...")
        models = {
            'category': joblib.load('models/category_model.pkl'),
            'priority': joblib.load('models/priority_model.pkl'),
            'sentiment': joblib.load('models/sentiment_model.pkl')
        }
        print("Models loaded!")
        interactive_predict(models)

    elif args.evaluate:
        df = load_data()
        print("Loading trained models...")
        models = {
            'category': joblib.load('models/category_model.pkl'),
            'priority': joblib.load('models/priority_model.pkl'),
            'sentiment': joblib.load('models/sentiment_model.pkl')
        }

        # Evaluate on test set
        X = df['processed_text']

        for name, target_col in [('category', 'category'), ('priority', 'priority'), ('sentiment', 'sentiment')]:
            y = df[target_col]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42, stratify=y
            )
            y_pred = models[name].predict(X_test)
            print(f"\n{name.upper()} MODEL:")
            print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
            print(classification_report(y_test, y_pred))

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
