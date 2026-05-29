import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

df = pd.read_csv("Week1\\Day2\\iMDB_dataset\\IMDB_Dataset.csv")

#Loading the Data to Head
#print(df.head())

#Inspecting the Data
print(df.shape)

#Check for null values
print(df.isnull().sum())

#Converting Labels into Numbers
df['sentiment'] = df['sentiment'].map({
    'positive': 1,
    'negative': 0
})

#Defining Features & labels
X = df['review']
y = df['sentiment']

#Train Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

#Converting Text to Vectors
from sklearn.feature_extraction.text import TfidfVectorizer

vectorizer = TfidfVectorizer(
    stop_words='english',
    max_features=5000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

#Create Logistic Regression Model
from sklearn.linear_model import LogisticRegression

model = LogisticRegression(max_iter=1000)

#Train the Model
model.fit(X_train_vec, y_train)

#Predicting the Test Set
y_pred = model.predict(X_test_vec)

#Evaluate the Model
from sklearn.metrics import accuracy_score

accuracy = accuracy_score(y_test, y_pred)

print("Accuracy:", accuracy)

#Detailed Metrics
from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))

#Predicting a New Review
new_review = [
    "The movie was fantastic and the acting was amazing"
]

new_review_vec = vectorizer.transform(new_review)

prediction = model.predict(new_review_vec)

print("Prediction Score: ", prediction)

if prediction==1:
    print("The review is Positive")
else:
    print("The review is Negative")



ConfusionMatrixDisplay.from_predictions(y_test, y_pred)

plt.title("Sentiment Classification Results")
plt.show()


#Working Pipeline

""" CSV Dataset (50,000 Reviews)
                │
                ▼
        Load using Pandas
                │
                ▼
     Positive/Negative Labels
                │
                ▼
        Train-Test Split
      40,000      10,000
                │
                ▼
      TF-IDF Vectorization
                │
                ▼
      Numerical Feature Matrix
                │
                ▼
       Logistic Regression
                │
                ▼
          Predictions
                │
                ▼
 Accuracy / Precision / Recall """