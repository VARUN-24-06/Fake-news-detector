import pandas as pd

# Load the CSV files
fake_df = pd.read_csv("Fake.csv")
true_df = pd.read_csv("True.csv")

# Add labels: 0 = Fake, 1 = Real
fake_df["label"] = 0
true_df["label"] = 1

# Combine both datasets into one
data = pd.concat([fake_df, true_df])

# Shuffle the data
data = data.sample(frac=1).reset_index(drop=True)

# Display first few rows
print("First 5 rows of the combined dataset:")
print(data.head())

# Show label distribution
print("\nLabel distribution (0 = Fake, 1 = Real):")
print(data["label"].value_counts())

from sklearn.feature_extraction.text import TfidfVectorizer

# Use only the necessary columns
texts = data['text']      # or 'content' if the column is named that
labels = data['label']

import string

def clean_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text

data['text'] = data['text'].apply(clean_text)

# Create a TF-IDF Vectorizer
vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)

# Convert the text data into TF-IDF features
X = vectorizer.fit_transform(texts)
y = labels

print("Text vectorization complete!")
print("TF-IDF matrix shape:", X.shape)
from sklearn.model_selection import train_test_split

# Split the data (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("Train-test split complete!")
print("Training data shape:", X_train.shape)
print("Testing data shape:", X_test.shape)

from sklearn.linear_model import LogisticRegression

# Create the model
model = LogisticRegression()

# Train the model
model.fit(X_train, y_train)

print("Model training complete!")

print("Example prediction (0 = Fake, 1 = Real):", model.predict(X_test[0]))

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Predict on test set
y_pred = model.predict(X_test)

# Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\nAccuracy: {acc:.4f}")

# Classification Report
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=["Fake", "Real"]))

# Confusion Matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

import pickle

# Save the trained model
with open("model.pkl", "wb") as model_file:
    pickle.dump(model, model_file)

# Save the TF-IDF vectorizer
with open("vectorizer.pkl", "wb") as vec_file:
    pickle.dump(vectorizer, vec_file)

print("Model and vectorizer saved!")