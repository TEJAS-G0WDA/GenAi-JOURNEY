#Movie Review Sentiment Classifier

import pandas as pd #Handle dataset

from sklearn.model_selection import train_test_split #Split training/testing data
from sklearn.feature_extraction.text import CountVectorizer #Convert text into numbers
from sklearn.linear_model import LogisticRegression #Classification algorithm
from sklearn.metrics import accuracy_score, classification_report #Measure accuracy, F1-score

#Sample Mock Dataset
data = {
    'review': [
        'Amazing movie',
        'Excellent acting',
        'Loved every scene',
        'Fantastic film',
        'Great story',
        'Worst movie ever',
        'Terrible acting',
        'Very boring',
        'Waste of time',
        'Bad storyline'
    ],

    'sentiment': [
        1, 1, 1, 1, 1,
        0, 0, 0, 0, 0
    ]
}

df = pd.DataFrame(data)

print(df)

#Seperate Features & Labels
X = df['review']
y = df['sentiment']

#Converting Text to Numbers (because ML Models cannot undertstand Text Directly)
vectorizer = CountVectorizer()

X_vectorized = vectorizer.fit_transform(X)

#Split Data into Training & Testing Sets (80/20 Rule)
X_train, X_test, y_train, y_test = train_test_split(
    X_vectorized,
    y,
    test_size=0.2,
    random_state=42
)

#Create & Train Logistic Regression Model
model = LogisticRegression()