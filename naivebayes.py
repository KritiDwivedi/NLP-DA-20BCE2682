#NAIVE BAYES:
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset


# Load the Twitter Sentiment Analysis dataset
dataset = load_dataset('carblacac/twitter-sentiment-analysis')
df = dataset['train'].to_pandas()


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['feeling'], test_size=0.2, random_state=42)


# Vectorize the text using CountVectorizer
vectorizer = CountVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train a Naive Bayes classifier
naive_bayes = MultinomialNB()
naive_bayes.fit(X_train_vec, y_train)


# Predictions
y_pred = naive_bayes.predict(X_test_vec)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
