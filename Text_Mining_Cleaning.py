#Cleaning processes: TAKES 30MINUTES
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
os.chdir("/Users/uhoenig/Dropbox/Studium 2015-16 GSE/Term 3/Text Mining/Project/Data")
#+ [enchant.Dict('en')]

#Import the downloaded Documents
def read(name):
    with open(name) as data_file:
        data = json.load(data_file)
    return data

viral_articles_unclean = read("Documents_viral.txt")
nonviral_articles_unclean = read("Documents_nonviral.txt")

def cleanupDoc(s):

    s = str(re.sub('[^a-zA-Z]+',' ', s))
    d = enchant.Dict("en_US")
    tokens = nltk.word_tokenize(s)
    boolean = map(d.check, tokens)
    l = []
    for idx in range(len(tokens)):
    	if boolean[idx]:
    		l.append(tokens[idx])
    cleanup = nltk.word_tokenize(unicode(" ".join(l),"utf-8"))
    stopset = set(stopwords.words('english')+ [string.punctuation]) 
    cleanup = [token.lower() for token in cleanup if token.lower() not in stopset and  len(token)>2]
    
    cleanup = np.array([PorterStemmer().stem(t) for t in cleanup])
    cleanup = [token.lower() for token in cleanup if token.lower() not in stopset and  len(token)>2]
    return cleanup

viral_articles_clean = []
viral_articles_clean_tokens = []
for i in range(len(viral_articles_unclean)):
    #For Text
    print i,"viral" 
    viral_articles_clean.append(' '.join(cleanupDoc(viral_articles_unclean[i]))) 
    #For tokens
    viral_articles_clean_tokens.append(cleanupDoc(viral_articles_unclean[i]))

viral_articles_clean = map(str,viral_articles_clean)    

nonviral_articles_clean = []
nonviral_articles_clean_tokens = []
for i in range(len(nonviral_articles_unclean)):
    #For text
    print i,"non_viral"
    nonviral_articles_clean.append(' '.join(cleanupDoc(nonviral_articles_unclean[i])))
    #For tokens
    nonviral_articles_clean_tokens.append(cleanupDoc(nonviral_articles_unclean[i]))
    
nonviral_articles_clean = map(str,nonviral_articles_clean)


def save(name,file):
    with open(name, 'w') as texts:
        json.dump( file, texts, indent = 3 )
    print "Dumped File"

save("Documents_nonviral_clean.txt",nonviral_articles_clean)
save("Documents_viral_clean.txt",viral_articles_clean)
save("Documents_viral_clean_tokens.txt",viral_articles_clean_tokens)
save("Documents_nonviral_clean_tokens.txt",nonviral_articles_clean_tokens)


