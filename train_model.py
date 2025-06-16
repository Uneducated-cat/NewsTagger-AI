import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import pickle
import json

# Load dataset
df = pd.read_csv("BBC News Train.csv")

# Preprocess text
def preprocess_text(text):
    """Cleans and prepares text data."""
    text = str(text).lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)  # Keep only alphabetic characters and spaces
    text = ' '.join(text.split())  # Remove extra whitespace
    return text

df['processed_text'] = df['Text'].apply(preprocess_text)

# Encode labels
label_encoder = LabelEncoder()
df['category_encoded'] = label_encoder.fit_transform(df['Category'])

# Vectorize text using TF-IDF
vectorizer = TfidfVectorizer(
    max_features=10000,
    stop_words='english',
    ngram_range=(1, 2),
    min_df=2,
    max_df=0.95
)
X = vectorizer.fit_transform(df['processed_text'])
y = df['category_encoded']

# Split data to evaluate model accuracy
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# --- Model Training and Evaluation ---
models = {
    'Naive Bayes': MultinomialNB(),
    'SVM': SVC(kernel='linear', probability=True, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42)  # Added model
}

model_accuracies = {}

print("Starting model training and evaluation...")
for name, model in models.items():
    # Train on the training subset
    print(f"Training {name}...")
    model.fit(X_train, y_train)

    # Evaluate on the test subset
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    model_accuracies[name] = accuracy
    print(f"  > {name} Accuracy: {accuracy:.4f}")

    # IMPORTANT: Retrain the model on the FULL dataset before saving for deployment
    model.fit(X, y)
    
    # Save the fully trained model
    with open(f'model_{name.lower().replace(" ", "_")}.pkl', 'wb') as f:
        pickle.dump(model, f)

# --- Save Supporting Files ---
# Save the vectorizer
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(vectorizer, f)

# Save the label encoder
with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# Save the calculated accuracies to a JSON file
with open('model_accuracies.json', 'w') as f:
    json.dump(model_accuracies, f)

print("\n✅ Models, vectorizer, label encoder, and accuracies have been saved successfully.")