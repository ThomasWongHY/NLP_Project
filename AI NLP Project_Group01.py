import os
import pandas as pd

#1 Load the data into a pandas data frame.
path = "/Users/thomas/Documents/Centennial Courses/COMP 237_Introduction to AI/NLP Project/"
filename = 'Youtube01-Psy.csv'
fullpath = os.path.join(path,filename)

youtube_data = pd.read_csv(filename)

#2 Carry out some basic data exploration and present your results. 
# (Note: You only need two columns for this project, make sure you identify them correctly, if any doubts ask your professor)
print("\nThe shape of dataset:")
print(youtube_data.shape)
print("\nThe information of dataset:")
print(youtube_data.info())
print("\nThe first 5 rows of dataframe:")
print(youtube_data.head())

youtube_comment = youtube_data[["CONTENT", "CLASS"]]
print("\nNumber of contents and unique values by classes:")
print(youtube_comment.groupby('CLASS').describe())
print("\nProportion of class")
print(youtube_comment['CLASS'].value_counts(normalize=True))

#3 Using nltk toolkit classes and methods prepare the data for model building, refer to the third lab tutorial in module 11
# Data Preprocessing
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer

nltk.download('punkt')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('stopwords')

import re

def preprocess(comments):

  # Convert to lowercase, Retain alphabet ONLY
  comments = " ".join(re.findall("[A-Za-z]+", comments.lower()))
  # Tokenization
  tokens = word_tokenize(comments)
  # Remove stop words
  words = [i for i in tokens if i not in stopwords.words("english")]
  # Stemming
  stemmer = PorterStemmer()
  words = [stemmer.stem(word) for word in words]
  # # Lemmatization
  # wordnet_lemmatizer = WordNetLemmatizer()
  # words = [wordnet_lemmatizer.lemmatize(word) for word in words]

  comments = " ".join(words)
  
  return comments

# Display the first 5 rows of new dataframe
youtube_comment['cleaned'] = youtube_comment['CONTENT'].apply(preprocess)
print("\nThe first 5 rows of new dataframe:")
print(youtube_comment.head())

# Count the numbe of each word in spam class (CLASS = 1)
spam_word_count = youtube_comment[youtube_comment['CLASS'] == 1]['cleaned'].str.split(expand=True).stack().value_counts().reset_index()
spam_word_count.columns = ['Word', 'Frequency'] 
print(spam_word_count)

import matplotlib.pyplot as plt
from wordcloud import WordCloud
#WordCloud to see the frequently used words
spam_words = ' '.join(list(youtube_comment[youtube_comment['CLASS'] == 1]['cleaned']))
spam_wc = WordCloud(width = 512,height = 512).generate(spam_words)
plt.figure(figsize = (10, 8), facecolor = 'k')
plt.imshow(spam_wc)
plt.axis('off')
plt.tight_layout(pad = 0)
plt.show()

#4 Present highlights of the output (initial features) such as the new shape of the data and any other useful information before proceeding.
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(youtube_comment['cleaned'])

# Print information of initial features
print("\nFeatures:")
print(count_vectorizer.get_feature_names_out())
print("\nDimensions of initial features:")
print(train_tc.shape)
print("\nType of initial features:")
print(type(train_tc))

#5 Downscale the transformed data using tfidf and again present highlights of the output (final features) 
# such as the new shape of the data and any other useful information before proceeding.
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)

# Print information of final features
print("\nDimensions of final features:")
print(train_tfidf.shape)
print("\nType of final features:")
print(type(train_tfidf))

#6 Use pandas.sample to shuffle the dataset, set frac = 1 
youtube_shuffled = youtube_comment.sample(frac=1)

#7 Using pandas split your dataset into 75% for training and 25% for testing, make sure to separate the class from the feature(s). 
# (Do not use test_train_split)
training_data = youtube_shuffled.sample(frac = 0.75)
testing_data = youtube_shuffled.drop(training_data.index)

print("\nNumber of training data")
print(len(training_data))
print("\nNumber of testing data")
print(len(testing_data))

#8 Fit the training data into a Naive Bayes classifier. 
count_vectorizer = CountVectorizer()
train_tc = count_vectorizer.fit_transform(training_data['cleaned'])
tfidf = TfidfTransformer()
train_tfidf = tfidf.fit_transform(train_tc)
classifier = MultinomialNB().fit(train_tfidf, training_data['CLASS'])

#9 Cross validate the model on the training data using 5-fold and print the mean results of model accuracy.
from sklearn.model_selection import cross_val_score

num_folds = 5
training_scores = cross_val_score(classifier, train_tfidf, training_data['CLASS'], cv=num_folds)

# Print the accuracy of each fold for training data
print("\nAccuracy of each fold in training data: ")
print(training_scores)

# Print the mean accuracy of all 5 folds for training data
print("\nMean Accuracy in training data: ")
print(training_scores.mean())

#10 Test the model on the test data, print the confusion matrix and the accuracy of the model.
test_tc = count_vectorizer.transform(testing_data['cleaned'])
test_tfidf = tfidf.transform(test_tc)

from sklearn.metrics import confusion_matrix, accuracy_score

test_predictions = classifier.predict(test_tfidf)

print("\nTest Predictions:")
print(test_predictions)
testing_scores = cross_val_score(classifier, test_tfidf, test_predictions, cv=num_folds)

# Print the accuracy of each fold for training data
print("\nAccuracy of each fold for testing data: ")
print(testing_scores)

# Print the mean accuracy of all 5 folds for training data
print("\nMean Accuracy for testing data: ")
print(testing_scores.mean())

# Print the confusion matrix
print("\nConfusion Matrix: ")
print(confusion_matrix(testing_data['CLASS'], test_predictions))

#11 As a group come up with 6 new comments (4 comments should be non spam and 2 comment spam) and pass them to the classifier and check the results. 
# You can be very creative and even do more happy with light skin tone emoticon.
new_comments = pd.DataFrame({'CONTENT': ["SO SICK OF THIS SONG!!! Can't stop listening this incredible song from youtube", 
                                         "I love this song very much", 
                                         "Guys, please check this music video in youtube and facebook", 
                                         "Get addicted to it! This will be the trillions'video in the future", 
                                         "please subscrible centennial ai channel and follow ours social media", 
                                         "It's party time, can't stop shaking my body when listening to this song!!!"],
                             'CLASS': [0, 0, 1, 0, 1, 0]})

new_comments_tc = count_vectorizer.transform(new_comments['CONTENT'])
new_comments_tfidf = tfidf.transform(new_comments_tc)

# Print prediction by classifier
new_comments_predictions = classifier.predict(new_comments_tfidf)
print("Predictions of new_comments:")
print(new_comments_predictions, '\n')

# Print the accuracy of the model
print("Accuracy of the model: ")
print(accuracy_score(new_comments['CLASS'], new_comments_predictions))

# Define the category map
category_map = {1: 'Spam', 0: 'No Spam'}

# Print the outputs
for sent, category in zip(new_comments['CONTENT'], new_comments_predictions):
    print('\nInput:', sent, '\nPredicted category:', \
            category_map[category])