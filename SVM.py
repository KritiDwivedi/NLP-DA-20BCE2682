#SVM
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from datasets import load_dataset
from sklearn.preprocessing import LabelEncoder


# Load the Twitter Sentiment Analysis dataset
dataset = load_dataset('carblacac/twitter-sentiment-analysis')
df = dataset['train'].to_pandas()


# Check the column names in your DataFrame
print(df.columns)


# Assuming the target column is named 'label' (replace with the correct column name)
text_column = 'text'  # Replace with the correct column name
target_column = 'feeling'  # Replace with the correct column name


# Ensure 'label' is categorical
label_encoder = LabelEncoder()
df['feeling'] = label_encoder.fit_transform(df['feeling'])


# Split the dataset
X_train, X_test, y_train, y_test = train_test_split(df[text_column], df[target_column], test_size=0.2, random_state=42)


# Use TfidfVectorizer instead of CountVectorizer
vectorizer = TfidfVectorizer()
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train an SVM classifier
svm = SVC(kernel='linear', C=3.0, probability=False, random_state=2)
svm.fit(X_train_vec, y_train)


# Predictions
y_pred = svm.predict(X_test_vec)


# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")
print(classification_report(y_test, y_pred))
