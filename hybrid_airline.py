import numpy as np
import pandas as pd
import sklearn.metrics as mt
from tabulate import tabulate
from sklearn.svm import LinearSVC
import sklearn.model_selection as ms
from matplotlib import pyplot as plt
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import Normalizer
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

# load data
filename = './airline.csv'
names = ['review.text','sentiment']
fields = ['review.text','sentiment']
review = pd.read_csv(filename, names=names, usecols=fields)

# Remove rows with 'neutral' sentiment
review_df = review[review['sentiment'] != 'neutral']

# split data 70% train 30% test
print('split data --- start')
array = review_df.values  # Use review_df here
X = array[:, 0:1]
Y = array[:, 1]
size = 0.3

# testX and trainX is review.tokened
# testY and trainY is sentiment
trainX, testX, trainY, testY = ms.train_test_split(X, Y, test_size=size, shuffle=True)
print('split data --- end')
print(testY)
print()

# feature extraction
print('feature extraction --- start')
tv = TfidfVectorizer(use_idf=True, min_df=0.0, max_df=1.0, ngram_range=(1, 2), sublinear_tf=False,
                     max_features=10000, smooth_idf=True, stop_words='english')
tv_train = tv.fit_transform(trainX.ravel())
tv_test = tv.transform(testX.ravel())  # transform test review into features
print('feature extraction --- end')

print()

# latent semantic analysis on train data (lexicon based)
print('latent semantic analysis --- start')
svd = TruncatedSVD(100)
lsa = make_pipeline(svd, Normalizer(copy=False))
lsa_train = lsa.fit_transform(tv_train)
lsa_test = lsa.transform(tv_test)  # transform tfidf test
print('latent semantic analysis --- end')

print()

# build model and predict with lsa train (machine learning)
print('build and predict --- start')
svm = LinearSVC()
svm.fit(lsa_train, trainY)
svm_pred = svm.predict(lsa_test)
print('build and predict --- end')

print()

# evaluation
print('\nModel Evaluation:')
svm_accuracy = np.round(mt.accuracy_score(testY, svm_pred), 3)
svm_precision = np.round(mt.precision_score(testY, svm_pred, average='macro'), 3)
svm_recall = np.round(mt.recall_score(testY, svm_pred, average='macro'), 3)
svm_f1 = np.round(mt.f1_score(testY, svm_pred, average='macro'), 3)

svm_metrics = np.array([svm_accuracy, svm_precision, svm_recall, svm_f1])
svm_metrics = pd.DataFrame([svm_metrics], columns=['accuracy', 'precision', 'recall', 'f1'], index=['metrics'])
print('Performance Metrics:')
print(tabulate(svm_metrics, headers='keys', tablefmt='github'))

# visualization
fig = plt.figure()
ax = svm_metrics.plot.bar()
plt.title('Hybrid Approach Performance Evaluation\n')
plt.ylabel('result')
plt.xlabel('model evualtion')
plt.xticks(rotation=-360)  # rotate x labels
plt.ylim([0.1, 1.0])
for item in ax.patches:  # show value on plot
    ax.annotate(np.round(item.get_height(), decimals=2), (item.get_x() + item.get_width() / 2., item.get_height()),
                ha='center', va='center', xytext=(0, 10), textcoords='offset points')
plt.savefig('./airlineresults/hybrid_performance.png', format='png', transparent=False)  # save result
plt.show()

print('\nConfusion Matrix of Hybrid Approach:\n')

# display and plot confusion matrix
labels = ['positive', 'negative']
svm_cm = mt.confusion_matrix(testY, svm_pred, labels=labels)

# plot
# display and plot confusion matrix
fig = plt.figure()
ax = fig.add_subplot(111)
plt.title('Confusion Matrix of Hybrid Approach\n')
fig.colorbar(ax.matshow(svm_cm))
ax.set_xticklabels([''] + labels)
ax.set_yticklabels([''] + labels)
plt.xlabel('predicted')
plt.ylabel('true')
plt.savefig('./airlineresults/hybrid_confusion_matrix.png',  format='png', transparent=False)  # save result
plt.show()

# display in table format
# level = [len(labels)*[0], list(range(len(labels)))]
# svm_cmf = pd.DataFrame(data=svm_cm,
#                        columns=pd.MultiIndex(levels=[['predicted:'], labels], labels=level),
#                        index=pd.MultiIndex(levels=[['actual:'], labels], labels=level))
# print(svm_cmf)
level = list(range(len(labels)))
columns = pd.MultiIndex.from_product([['predicted'], labels], names=['', ''])
index = pd.MultiIndex.from_product([['actual'], labels], names=['', ''])
svm_cmf = pd.DataFrame(data=svm_cm, columns=columns, index=index)
print(svm_cmf)


print('\nClassification Report of Hybrid Approach to Sentiment Analysis:\n')

# classification report for hybrid approach
svm_report = mt.classification_report(testY, svm_pred, labels=labels)
print(svm_report)

print()

# output performance to csv
svm_metrics.to_csv('./airlineresults/hybrid_performance_result.csv', index=None, header=True)

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.svm import LinearSVC
from sklearn.pipeline import make_pipeline

# Preprocess your sentence (replace 'your_sentence' with your actual sentence)
your_sentence1 = "This was one of my worst flight experiences"

# Transform your sentence using TF-IDF vectorizer
your_sentence_transformed1 = tv.transform([your_sentence1])

# Apply Latent Semantic Analysis (LSA)
your_sentence_lsa1 = lsa.transform(your_sentence_transformed1)

# Predict sentiment using the SVM model
predicted_sentiment1 = svm.predict(your_sentence_lsa1)[0]

# Preprocess your sentence (replace 'your_sentence' with your actual sentence)
your_sentence2 = "the flight was great. the cabin crew was very helpful."

# Transform your sentence using TF-IDF vectorizer
your_sentence_transformed2 = tv.transform([your_sentence2])

# Apply Latent Semantic Analysis (LSA)
your_sentence_lsa2 = lsa.transform(your_sentence_transformed2)

# Predict sentiment using the SVM model
predicted_sentiment2 = svm.predict(your_sentence_lsa2)[0]

print('Review: ',your_sentence1)
# Print the predicted sentiment
if predicted_sentiment1 == 'positive':
    print("The model predicts a positive sentiment.")
elif predicted_sentiment1 == 'negative':
    print("The model predicts a negative sentiment.")
elif predicted_sentiment1 == 'neutral':
    print("The model predicts a neutral sentiment.")
else:
    print("The model predicts an unknown sentiment.")


print('Review: ',your_sentence2)
# Print the predicted sentiment
if predicted_sentiment2 == 'positive':
    print("The model predicts a positive sentiment.")
elif predicted_sentiment2 == 'negative':
    print("The model predicts a negative sentiment.")
elif predicted_sentiment2 == 'neutral':
    print("The model predicts a neutral sentiment.")
else:
    print("The model predicts an unknown sentiment.")

# end of hybrid approach