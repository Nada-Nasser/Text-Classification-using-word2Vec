import gensim as gensim
import tokenizer as tokenizer
from nltk import RegexpTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
import os
from sklearn import svm, metrics


def read_file_in_dir(directory):
    docs = []
    with os.scandir(directory) as entries:
        for entry in entries:
            file = open(directory + "/" + entry.name, 'r')
            data = file.read().replace("\n", "")
            docs.append(data)
    return docs


list_of_zeros = [0] * 1000
list_of_ones = [1] * 1000

y_data = list_of_ones + list_of_zeros

neg_docs = read_file_in_dir("neg/")
pos_docs = read_file_in_dir("pos/")
all_docs = pos_docs + neg_docs

x_train, x_test, y_train, y_test = train_test_split(all_docs, y_data ,train_size=10)

token = RegexpTokenizer("[\w']+")
all_words = []
for reviews in all_docs:
     all_words.append(token.tokenize(reviews))

model = gensim.models.Word2Vec(all_words,
                               vector_size=150,
                               window=10,
                               min_count= 5,
                               workers=10)

print(model.wv["happy"])
print(model)

all_reviews = []
for review in x_train:
    words = token.tokenize(review)
    avgs = [0 for i in range(150)]
    for w in words:
        if w in model.wv.key_to_index:
            vec = model.wv[w]
            for i in range(0, 150):
                avgs[i] += vec[i] / len(words)
        else:
            print(w)
    all_reviews.append(avgs)

# # Create a svm Classifier
clf = svm.SVC(kernel='linear')  # Linear Kernel

# Train the model using the training sets
clf.fit(all_reviews, y_train)

# # Predict the response for test dataset
# tfs = Tfidf_Vectorizer.transform(x_test).astype('float64')
y_pred = clf.predict(all_reviews)

print("Accuracy:", metrics.accuracy_score(y_train, y_pred))

# # tfs = Tfidf_Vectorizer.transform(["Iam very Sad"]).astype('float64')
# # print(clf.predict(tfs))
