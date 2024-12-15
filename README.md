# Sentiment Analysis with Logistic Regression and BERT

This project uses the Sentiment140 dataset to perform sentiment analysis with Logistic Regression and BERT.

## Setup

### Kaggle API:
Place `kaggle.json` in `~/.kaggle/` and set permissions:
<pre>
<code>
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
</code>
</pre>

### Download Dataset:
<pre>
<code>
!kaggle datasets download -d kazanova/sentiment140
</code>
</pre>

### Extract Data:
<pre>
<code>
from zipfile import ZipFile
with ZipFile('/content/sentiment140.zip', 'r') as zip:
    zip.extractall()
</code>
</pre>

## Preprocessing

Load dataset, rename columns, replace target values (`4` → `1`), and split data:
<pre>
<code>
import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')
data['target'].replace({4: 1}, inplace=True)
data = data[['text', 'target']]
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, stratify=data['target'], random_state=42)
</code>
</pre>

## Logistic Regression Model

Preprocess text and train the Logistic Regression model:
<pre>
<code>
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer()
X_train_lr = vectorizer.fit_transform(X_train)
X_test_lr = vectorizer.transform(X_test)

logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train_lr, y_train)
</code>
</pre>

## BERT Model

Preprocess data for BERT and train the model:
<pre>
<code>
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import DataLoader

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
train_loader = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset_bert, batch_size=16, shuffle=False)

bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
optimizer = AdamW(bert_model.parameters(), lr=2e-5)
</code>
</pre>

## Results

Compare the accuracy of both models:
<pre>
<code>
print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.2f}")
print(f"BERT Test Accuracy: {bert_test_accuracy:.2f}")
</code>
</pre>

## Save BERT Model:
<pre>
<code>
bert_model.save_pretrained('/content/bert_sentiment_model')
bert_tokenizer.save_pretrained('/content/bert_sentiment_model')
</code>
</pre>

This project compares Logistic Regression and BERT for sentiment analysis, with BERT outperforming Logistic Regression.
