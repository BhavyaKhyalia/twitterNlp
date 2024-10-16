# Twitter Sentiment Analysis

This project uses the Sentiment140 dataset for sentiment analysis with Logistic Regression.

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

Load dataset, rename columns, replace target values (`4` → `1`), and apply stemming:
<pre>
<code>
twitter_data = pd.read_csv('file.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')
twitter_data.replace({'target': {4: 1}}, inplace=True)
</code>
</pre>

## Model Training

Convert text to numerical data using `TfidfVectorizer`, split data, and train a Logistic Regression model:
<pre>
<code>
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)
</code>
</pre>

## Results

Compute accuracy on training and test sets.



