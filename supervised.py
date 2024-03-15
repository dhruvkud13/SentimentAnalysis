import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.utils import to_categorical

# Load your dataset
# Replace 'your_dataset.csv' with your dataset file path
df = pd.read_csv('airline_data.csv')

# Filter out tweets with neutral sentiment
df = df[df['sentiment'] != 'neutral']

# Split your filtered data into training and testing sets
X = df['tweet']
y = df['sentiment']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Tokenize and preprocess text data
max_words = 5000
tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(X_train)

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

max_sequence_length = 200
X_train_pad = pad_sequences(X_train_seq, maxlen=max_sequence_length)
X_test_pad = pad_sequences(X_test_seq, maxlen=max_sequence_length)

# Encode labels
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

y_train_categorical = to_categorical(y_train_encoded)
y_test_categorical = to_categorical(y_test_encoded)

# Build LSTM model
embedding_dim = 50

model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_sequence_length))
model.add(LSTM(128, return_sequences=True))
model.add(Dropout(0.2))
model.add(LSTM(128))
model.add(Dense(3, activation='softmax'))  # Assuming 3 classes: positive, negative, neutral

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
print(model.summary())

# Train the model
epochs = 2
batch_size = 32

history = model.fit(X_train_pad, y_train_categorical, epochs=epochs, batch_size=batch_size,
                    validation_split=0.2, verbose=1)

y_pred = model.predict(X_test_pad)

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
plt.savefig('confusion_matrix_hybrid1.png', format='png', transparent=False)

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
# Evaluate the model
loss, accuracy = model.evaluate(X_test_pad, y_test_categorical, verbose=0)
print(f'Test loss: {loss:.4f}, Test accuracy: {accuracy:.4f}')

# Plot training history
plt.figure(figsize=(8, 6))
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc='lower right')
plt.show()

# Test a sentence with the trained model
test_sentence = ["airlines was shit"]
test_sentence_seq = tokenizer.texts_to_sequences(test_sentence)
test_sentence_pad = pad_sequences(test_sentence_seq, maxlen=max_sequence_length)
test_sentence_pred = model.predict(test_sentence_pad)

predicted_sentiment = label_encoder.inverse_transform(np.argmax(test_sentence_pred, axis=-1))
print(f"Predicted sentiment for test sentence: {predicted_sentiment[0]}")
