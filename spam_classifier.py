import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -------------------------------
# 1. LOAD DATA
# -------------------------------
df = pd.read_csv("spam.csv", encoding='latin-1')[['v1', 'v2']]
df.columns = ['label', 'text']
df['label'] = df['label'].map({'spam': 1, 'ham': 0})

print(df.head())

# -------------------------------
# 2. SPLIT DATA
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    df['text'], df['label'], test_size=0.2, random_state=42)

# -------------------------------
# 3. TEXT VECTORIZATION (TF-IDF)
# -------------------------------
vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# -------------------------------
# 4. TRAIN NAIVE BAYES MODEL
# -------------------------------
nb_model = MultinomialNB()
nb_model.fit(X_train_vec, y_train)
nb_pred = nb_model.predict(X_test_vec)

# -------------------------------
# 5. TRAIN SVM MODEL
# -------------------------------
svm_model = LinearSVC()
svm_model.fit(X_train_vec, y_train)
svm_pred = svm_model.predict(X_test_vec)

# -------------------------------
# 6. EVALUATION
# -------------------------------
print("\n========== Naive Bayes Results ==========")
print("Accuracy:", accuracy_score(y_test, nb_pred))
print(classification_report(y_test, nb_pred))

print("\n========== SVM Results ==========")
print("Accuracy:", accuracy_score(y_test, svm_pred))
print(classification_report(y_test, svm_pred))

# -------------------------------
# 7. PREDICT ON NEW EMAILS
# -------------------------------
def predict_email(text):
    text_vec = vectorizer.transform([text])
    svm_result = svm_model.predict(text_vec)[0]
    return "SPAM" if svm_result == 1 else "HAM (not spam)"

print("\nTest prediction:")
print(predict_email("Congratulations! You've won a free iPhone!"))
print(predict_email("Hi John, please send me the report."))
