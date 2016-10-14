import textmining as tm
import numpy as np
import os
from nltk import word_tokenize
import string
import nltk
from nltk.corpus import stopwords 
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer
from nltk import PorterStemmer
import re
import string
import enchant
import json
import pandas
from wordcloud import WordCloud
import matplotlib.pyplot as plt

#This creates my document term matrix (after properly cleaned and stemmed)
def read(name):
    with open(name) as data_file:
        data = json.load(data_file)
    return data

#This creates my document term matrix (after properly cleaned and stemmed)
def read(name):
    with open(name) as data_file:
        data = json.load(data_file)
    return data

#Read in the data first
viral_articles_clean = read("Documents_viral_clean.txt")
nonviral_articles_clean = read("Documents_nonviral_clean.txt")
viral_articles_clean_tokens = read("Documents_viral_clean_tokens.txt")
nonviral_articles_clean_tokens = read("Documents_nonviral_clean_tokens.txt")

from copy import copy
labels = np.hstack((np.repeat(1, len(viral_articles_clean)), np.repeat(0, len(nonviral_articles_clean))))
docs = copy(viral_articles_clean)
docs.extend(nonviral_articles_clean)

#Average article length (300words)
tokenlength = map(len,viral_articles_clean_tokens)
avglength = sum(tokenlength)/len(tokenlength)

print avglength

#Get most frequent viral words / non viral words
from itertools import chain
all_viral_words = list(chain(*viral_articles_clean_tokens))
all_nonviral_words = list(chain(*nonviral_articles_clean_tokens))
most_viral_words = nltk.FreqDist(all_viral_words).most_common(10) 
most_nonviral_words = nltk.FreqDist(all_nonviral_words).most_common(10)
print most_viral_words
print most_nonviral_words


##DONT DIFFER AT ALL.......


import csv
#Get those words with highest tf-idf score:
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer(min_df=1)
X = vectorizer.fit_transform(viral_articles_clean)
idf = vectorizer.idf_
dc = dict(zip(vectorizer.get_feature_names(), idf))
import operator
          
sorted_x = sorted(dc.items(), key=operator.itemgetter(1))


print sorted_x[0:10]
#x = {1: 2, 3: 4, 4: 3, 2: 1, 0: 0}
#sorted_x = sorted(x.items(), key=operator.itemgetter(1), reverse = True)
#print sorted_x

with open('tfidfviral.csv','w') as out:
    csv_out=csv.writer(out)
    csv_out.writerow(['word','freq'])
    for row in sorted_x:
        csv_out.writerow(row)

import pandas as pd
from sklearn.cross_validation import train_test_split
from sklearn.metrics import roc_auc_score
data_raw = pd.DataFrame({'article': docs, 'label': labels})


data_raw.shape

docs_train, docs_test, y_train, y_test = train_test_split(data_raw[['article']], data_raw.label, )

countvectorizer = CountVectorizer(max_features = 20000, ngram_range = (1,2))
X_train_bow = countvectorizer.fit_transform(docs_train.article)
X_test_bow = countvectorizer.transform(docs_test.article)


from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression

LR = LogisticRegression()
LR.fit(X_train_bow, y_train)
preds_LR = LR.predict(X_test_bow)
preds_train_LR = LR.predict(X_train_bow)


accuracy_test = accuracy_score(y_test, preds_LR)
accuracy_train = accuracy_score(y_train, preds_train_LR)
print 'Test Accuracy {0} | Train Accuracy {1}'.format(accuracy_test, accuracy_train)
preds_LR_proba = LR.predict_proba(X_test_bow)
print roc_auc_score(y_test,preds_LR_proba[:,1])
print roc_auc_score(y_train,preds_train_LR)

from sklearn.feature_extraction.text import TfidfTransformer
transformer = TfidfTransformer()
X_train_tfidf = transformer.fit_transform(X_train_bow)
X_test_tfidf = transformer.transform(X_test_bow)

preds_LR_proba = LR.predict_proba(X_test_tfidf)

print roc_auc_score(y_test,preds_LR_proba[:,1])
print roc_auc_score(y_train,preds_train_LR_tfidf)




LR.fit(X_train_tfidf, y_train)
preds_LR_tfidf = LR.predict(X_test_tfidf)
preds_train_LR_tfidf = LR.predict(X_train_tfidf)
accuracy_test_tfidf = accuracy_score(y_test, preds_LR_tfidf)
accuracy_train_tfidf = accuracy_score(y_train, preds_train_LR_tfidf)
print 'Test Accuracy {0} | Train Accuracy {1}'.format(accuracy_test_tfidf, accuracy_train_tfidf)



features = countvectorizer.get_feature_names()
absolute_coefs = abs(LR.coef_)[0]
most_important_features = np.asarray(features)[np.argsort(absolute_coefs)[::-1][0:50]]
print most_important_features 

from sklearn.decomposition import TruncatedSVD
svd = TruncatedSVD(n_components = 110)
train_svd = svd.fit_transform(X_train_tfidf)
test_svd = svd.transform(X_test_tfidf)

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_jobs = 3, n_estimators = 100) 
# wo die dimensionalitaet kleiner is koennen wir den auch nutzen

rf.fit(train_svd, y_train)
preds_rf = rf.predict(test_svd)
preds_rf_train = rf.predict(train_svd)

LR.fit(train_svd, y_train)
preds_LR_svd = LR.predict(test_svd)
preds_LR_svd_train = LR.predict(train_svd)

accuracy_rf_train = accuracy_score(y_train, preds_rf_train)
accuracy_rf_test = accuracy_score(y_test, preds_rf)

accuracy_lr_svd_train = accuracy_score(y_train, preds_LR_svd_train)
accuracy_lr_svd_test = accuracy_score(y_test, preds_LR_svd)

#print 'Logistic Regression: Train {0} | Test {1}'.format(accuracy_lr_svd_train, accuracy_lr_svd_test)
#print 'Random Forest: Train {0} | Test {1}'.format(accuracy_rf_train, accuracy_rf_test)

#test LR on svd 
preds_LR_proba = LR.predict_proba(test_svd)
print roc_auc_score(y_test,preds_LR_proba[:,1])
#train LR on svd
preds_LR_proba = LR.predict_proba(train_svd)
print roc_auc_score(y_train,preds_LR_proba[:,1])
#print np.mean(preds_LR_proba[:,0])

########### rf
#test rf on svd 
preds_LR_proba = rf.predict_proba(test_svd)
print roc_auc_score(y_test,preds_LR_proba[:,1])
#train LR on svd 
preds_LR_proba = rf.predict_proba(train_svd)
print roc_auc_score(y_train,preds_LR_proba[:,1])
#print np.mean(preds_LR_proba[:,0])



from sklearn.decomposition import LatentDirichletAllocation
lda = LatentDirichletAllocation(n_topics = 10)

X_train_lda = lda.fit_transform(X_train_tfidf)
X_test_lda = lda.transform(X_test_tfidf)




LR.fit(X_train_lda, y_train)
preds_lr_lda = LR.predict(X_test_lda)
preds_lr_lda_train = LR.predict(X_train_lda)

acc_lr_lda = accuracy_score(y_test, preds_lr_lda)
acc_lr_lda_train = accuracy_score(y_train, preds_lr_lda_train)

print 'Test Accuracy {0} | Train Accuracy {1}'.format(acc_lr_lda, acc_lr_lda_train)
preds_LR_proba = LR.predict_proba(X_test_lda)
print roc_auc_score(y_test,preds_LR_proba[:,1])
print roc_auc_score(y_train,preds_lr_lda_train)


log_odds_topics = np.exp(LR.coef_)
print log_odds_topics

from nltk.tokenize import RegexpTokenizer
from nltk.stem.porter import PorterStemmer
from gensim import corpora, models
import gensim

dictionary = corpora.Dictionary(viral_articles_clean_tokens)
corpus = [dictionary.doc2bow(text) for text in viral_articles_clean_tokens]
ldamodel = gensim.models.ldamodel.LdaModel(corpus, num_topics=10, id2word = dictionary, passes=10)

print(ldamodel.print_topics(num_topics=10, num_words=4))

