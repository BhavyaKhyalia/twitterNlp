# Install necessary dependencies
!pip install transformers torch
!pip install kaggle
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
!kaggle datasets download -d kazanova/sentiment140

# extracting the compressed dataset
from zipfile import ZipFile
dataset = '/content/sentiment140.zip'

with ZipFile(dataset,'r') as zip:
  zip.extractall()
  print('The dataset is extracted')

# Import libraries
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertForSequenceClassification, AdamW

# Load NLTK stopwords
nltk.download('stopwords')
stop_words = set(stopwords.words('english'))

# Load Dataset
column_names = ['target', 'id', 'date', 'flag', 'user', 'text']
data = pd.read_csv('/content/training.1600000.processed.noemoticon.csv', names=column_names, encoding='ISO-8859-1')
data['target'].replace({4: 1}, inplace=True)
data = data[['text', 'target']]

# Split dataset for both models
X_train, X_test, y_train, y_test = train_test_split(data['text'], data['target'], test_size=0.2, stratify=data['target'], random_state=42)

############################################
# Logistic Regression Model Implementation #
############################################
# Preprocessing for Logistic Regression
port_stem = PorterStemmer()

def preprocess_text(text):
    text = re.sub('[^a-zA-Z]', ' ', text)
    text = text.lower()
    text = text.split()
    text = [port_stem.stem(word) for word in text if word not in stop_words]
    return ' '.join(text)

# Apply preprocessing
X_train_lr = X_train.apply(preprocess_text)
X_test_lr = X_test.apply(preprocess_text)

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer()
X_train_lr = vectorizer.fit_transform(X_train_lr)
X_test_lr = vectorizer.transform(X_test_lr)

# Train Logistic Regression Model
logistic_model = LogisticRegression(max_iter=10000)
logistic_model.fit(X_train_lr, y_train)

# Evaluate Logistic Regression Model
train_preds_lr = logistic_model.predict(X_train_lr)
test_preds_lr = logistic_model.predict(X_test_lr)
train_accuracy_lr = accuracy_score(y_train, train_preds_lr)
test_accuracy_lr = accuracy_score(y_test, test_preds_lr)
print(f"Logistic Regression Train Accuracy: {train_accuracy_lr:.2f}")
print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.2f}")

################################
# BERT Model Implementation #
################################
# Load BERT Tokenizer
bert_tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Define Dataset Class for BERT
class SentimentDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts.iloc[index]
        label = self.labels.iloc[index]
        encoding = self.tokenizer(
            text,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        return {
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Create DataLoaders
train_dataset_bert = SentimentDataset(X_train, y_train, bert_tokenizer)
test_dataset_bert = SentimentDataset(X_test, y_test, bert_tokenizer)

train_loader_bert = DataLoader(train_dataset_bert, batch_size=16, shuffle=True)
test_loader_bert = DataLoader(test_dataset_bert, batch_size=16, shuffle=False)

# Load Pre-trained BERT Model
bert_model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
bert_model.to(device)

# Define Optimizer
optimizer = AdamW(bert_model.parameters(), lr=2e-5)

# Train BERT Model
def train_bert_model(model, train_loader, optimizer, device, epochs=3):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in train_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(train_loader):.4f}")

# Evaluate BERT Model
def evaluate_bert_model(model, test_loader, device):
    model.eval()
    predictions, true_labels = [], []

    with torch.no_grad():
        for batch in test_loader:
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            preds = torch.argmax(logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    accuracy = accuracy_score(true_labels, predictions)
    return accuracy

# Train and Evaluate BERT
train_bert_model(bert_model, train_loader_bert, optimizer, device, epochs=3)
bert_test_accuracy = evaluate_bert_model(bert_model, test_loader_bert, device)
print(f"BERT Test Accuracy: {bert_test_accuracy:.2f}")

###################################
# Combined Model Comparisons #
###################################
print("Model Comparison:")
print(f"Logistic Regression Test Accuracy: {test_accuracy_lr:.2f}")
print(f"BERT Test Accuracy: {bert_test_accuracy:.2f}")

# Save BERT Model
bert_model.save_pretrained('/content/bert_sentiment_model')
bert_tokenizer.save_pretrained('/content/bert_sentiment_model')
