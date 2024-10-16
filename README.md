Twitter Sentiment Analysis
This project uses the Sentiment140 dataset for sentiment analysis with Logistic Regression.

Setup
Kaggle API:
Place kaggle.json in ~/.kaggle/:

bash
Copy code
!mkdir -p ~/.kaggle
!cp kaggle.json ~/.kaggle/
!chmod 600 ~/.kaggle/kaggle.json
Download Dataset:

bash
Copy code
!kaggle datasets download -d kazanova/sentiment140
Extract Data:

python
Copy code
from zipfile import ZipFile
with ZipFile('/content/sentiment140.zip', 'r') as zip:
    zip.extractall()
Preprocessing
Load dataset, rename columns, replace target values (4 → 1), and apply stemming:
python
Copy code
twitter_data = pd.read_csv('file.csv', names=['target', 'id', 'date', 'flag', 'user', 'text'], encoding='ISO-8859-1')
twitter_data.replace({'target': {4: 1}}, inplace=True)
Model Training
Convert text to numerical data using TfidfVectorizer, split data, and train a Logistic Regression model:
python
Copy code
from sklearn.linear_model import LogisticRegression
model = LogisticRegression(max_iter=10000)
model.fit(X_train, Y_train)
Results
Compute accuracy on training and test sets.
