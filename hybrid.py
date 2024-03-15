import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.pipeline import make_pipeline

# Load your dataset
# Replace 'your_dataset.csv' with your dataset file path
df = pd.read_csv('airline_data.csv')

# Filter out tweets with neutral sentiment
df = df[df['sentiment'] != 'neutral']

# Split your filtered data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(df['tweet'], df['sentiment'], test_size=0.3, random_state=42)

# Create TF-IDF features
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# Perform Latent Semantic Analysis (LSA) on the TF-IDF vectors
svd = TruncatedSVD(100)
lsa = make_pipeline(svd)
X_train_lsa = lsa.fit_transform(X_train_tfidf)
X_test_lsa = lsa.transform(X_test_tfidf)

# Train a LinearSVC classifier
clf = LinearSVC()
clf.fit(X_train_lsa, y_train)

# Predict sentiment on the test data
y_pred = clf.predict(X_test_lsa)

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
plt.savefig('confusion_matrix_hybrid2.png', format='png', transparent=False)

# Display the figure
plt.show()

level = list(range(len(labels)))
columns = pd.MultiIndex.from_product([['predicted'], labels], names=['', ''])
index = pd.MultiIndex.from_product([['actual'], labels], names=['', ''])
svm_cmf = pd.DataFrame(data=svm_cm, columns=columns, index=index)
print(svm_cmf)
# Print classification report
classification_rep = classification_report(y_test, y_pred)
print("Classification Report:")
print(classification_rep)

# Test a sentence with the trained model
test_sentence = "It was an unpleasant experience, the cabin crew were not helpful"
test_sentence_tfidf = vectorizer.transform([test_sentence])
test_sentence_lsa = lsa.transform(test_sentence_tfidf)
test_sentence_pred = clf.predict(test_sentence_lsa)

print("Review", test_sentence)
print(f"Predicted sentiment for test sentence: {test_sentence_pred[0]}")

# Test a sentence with the trained model
test_sentence1 = "Their business class is awesome! Totally worth it"
test_sentence_tfidf1 = vectorizer.transform([test_sentence1])
test_sentence_lsa1 = lsa.transform(test_sentence_tfidf1)
test_sentence_pred1 = clf.predict(test_sentence_lsa1)

print("Review", test_sentence1)
print(f"Predicted sentiment for test sentence: {test_sentence_pred1[0]}")