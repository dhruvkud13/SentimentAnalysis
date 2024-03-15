import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from nltk.corpus import sentiwordnet as swn
from nltk.corpus import wordnet as wn
from nltk.tokenize import word_tokenize
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer

# Load your dataset
# Replace 'your_dataset.csv' with your dataset file path
df = pd.read_csv('mobile_data.csv')

# Preprocess your data and create SentiWordNet scores
def sentiwordnet_score(text):
    tokens = word_tokenize(text)
    sentiment_score = 0
    for token in tokens:
        synsets = wn.synsets(token)
        if not synsets:
            continue
        synset = synsets[0]  # Take the first synset
        senti_synset = swn.senti_synset(synset.name())
        sentiment_score += senti_synset.pos_score() - senti_synset.neg_score()
    return sentiment_score

df['SentiWordNet_Score'] = df['tweet'].apply(sentiwordnet_score)

# Define a threshold for sentiment classification
threshold = 0  # You can adjust this threshold as needed

# Assign sentiment labels based on the threshold
df['Sentiment'] = df['SentiWordNet_Score'].apply(lambda score: 'positive' if score > threshold else 'negative')

# Split your data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['Sentiment'], test_size=0.3, random_state=42)

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Train a LinearSVC classifier
clf = LinearSVC()
clf.fit(X_train_tfidf, y_train)

# Predict sentiment on the test data
y_pred = clf.predict(X_test_tfidf)

confusion = confusion_matrix(y_test, y_pred)
labels = ['positive', 'negative']
# Save confusion matrix as an image
import sklearn.metrics as mt

# Calculate the confusion matrix
svm_cm = mt.confusion_matrix(y_test, y_pred, labels=labels)

# Create a figure and axis
fig = plt.figure()
ax = fig.add_subplot(111)

# Set the title and color bar
plt.title('Confusion Matrix of Hybrid Approach')
cax = ax.matshow(svm_cm)
fig.colorbar(cax)

# Set the tick labels for both axes
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)

# Label the axes
plt.xlabel('Predicted')
plt.ylabel('Actual')

# Save the figure
plt.savefig('confusion_matrix_unsupervised2.png', format='png', transparent=False)

# Display the figure
plt.show()

# Calculate and print the classification report
classification_rep = classification_report(y_test, y_pred)
print("\nClassification Report:")
print(classification_rep)

plt.show()

level = list(range(len(labels)))
columns = pd.MultiIndex.from_product([['predicted'], labels], names=['', ''])
index = pd.MultiIndex.from_product([['actual'], labels], names=['', ''])
svm_cmf = pd.DataFrame(data=svm_cm, columns=columns, index=index)
print(svm_cmf)

# ... (previous code)

# Input your test sentence
test_sentence1 = "The worst phone used ever"

# Preprocess the test sentence and create SentiWordNet scores
test_sentence_score1 = sentiwordnet_score(test_sentence1)

# Assign sentiment label based on the threshold
test_sentence_sentiment = 'positive' if test_sentence_score1 > threshold else 'negative'

# Transform the test sentence using the TF-IDF vectorizer
test_sentence_tfidf1 = vectorizer.transform([test_sentence1])

# Predict sentiment on the test sentence
test_sentence_pred1 = clf.predict(test_sentence_tfidf1)

# Print the test sentence and predicted sentiment
print("Review:", test_sentence1)
print("Predicted Sentiment:", test_sentence_pred1[0])

# Input your test sentence
test_sentence2 = "This is the phone with the best features"

# Preprocess the test sentence and create SentiWordNet scores
test_sentence_score2 = sentiwordnet_score(test_sentence2)

# Assign sentiment label based on the threshold
test_sentence_sentiment2 = 'positive' if test_sentence_score2 > threshold else 'negative'

# Transform the test sentence using the TF-IDF vectorizer
test_sentence_tfidf2 = vectorizer.transform([test_sentence2])

# Predict sentiment on the test sentence
test_sentence_pred2 = clf.predict(test_sentence_tfidf2)

# Print the test sentence and predicted sentiment
print("Review:", test_sentence2)
print("Predicted Sentiment:", test_sentence_pred2[0])

# # You can also save the trained model for future use
# from joblib import dump
# dump(clf, 'sentiment_model.joblib')
