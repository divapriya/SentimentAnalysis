from bs4 import BeautifulSoup
import cython
import random
import pandas as pd  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.cluster import KMeans
from gensim.models import word2vec
import nltk.data
from nltk.tokenize import sent_tokenize
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
from sklearn.naive_bayes import MultinomialNB
import numpy as np
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
import re    
import logging
import gensim

LabeledSentence = gensim.models.doc2vec.LabeledSentence
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# Load dataset 
train = pd.read_csv("C:\Users\PSarka\Desktop\sentimentanalysis\labeledTrainData.tsv",header=0,delimiter="\t", quoting=3)
test = pd.read_csv(r"C:\Users\PSarka\Desktop\sentimentanalysis\testData.tsv",header=0,delimiter="\t", quoting=2)
sample = pd.read_csv("C:\Users\PSarka\Desktop\sentimentanalysis\sampleSubmission.csv")


#Simple Processing splitting the reviews into sentences,in order to train doc2vec
#Proocessing or Cleaning Text
def remove_punctuations(text):
    wordlist=[]    
    sentences=sent_tokenize(text)
    for sentence in sentences:
        #words=word_tokenize(sentence)
        words=[word for word in sentence.split()]               
        words=[re.sub(r'[^\w\s]','',word) for word in words]     
        wordlist.append(words)   
    return wordlist

def processing(raw_review):
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    # 2. Convert all to lower Case
    review_text=review_text.lower()    
    # 3. Remove Punctuations        
    letters_only = remove_punctuations(review_text)
    return letters_only
    
clean_train_review=[]
for i in xrange( 0,1000):
    clean_train_review.append(processing(train["review"][i]))
    
clean_test_review=[]
for i in xrange( 0,10):
    clean_test_review.append(processing(test["review"][i]))

#Training Doc2Vec model
def labelizeReviews(reviews, label_type):
    labelized = []
    for review in reviews:    
        for i,v in enumerate(review):
            label = '%s_%s'%(label_type,i)
            print LabeledSentence(v, [label]) 
            labelized.append(LabeledSentence(v, [label]))
    #print labelized
    return labelized
    
train_sentences=labelizeReviews(clean_train_review,"SENT")
test_sentences=labelizeReviews(clean_test_review,"SENT")
size = 100

#instantiate our DM and DBOW models
model_dm = gensim.models.Doc2Vec(min_count=2, window=10, size=size, sample=1e-3, negative=5, workers=3)
#model_dbow = gensim.models.Doc2Vec(min_count=2, window=10, size=size, sample=1e-3, negative=5, dm=0, workers=3)

#build vocab over all reviews
model_dm.build_vocab(train_sentences)
#model_dbow.build_vocab(sentences)

#We pass through the data set multiple times, shuffling the training reviews each time to improve accuracy.
for epoch in range(10):
#    perm = np.random.permutation(sentences.shape[0])
     model_dm.train(train_sentences)
    #model_dbow.train(sentences[perm])
#model_dm.save("Doc2vec.bin")


#Get training set vectors from our models
def getVecs(model, corpus, size):           
    vecs = [np.array(model[z.labels[0]]).reshape((1, size)) for z in corpus]
    return np.concatenate(vecs)

train_vecs_dm = getVecs(model_dm,train_sentences, size)
#train_vecs_dbow = getVecs(model_dbow,sentences, size)

#train_vecs = np.hstack((train_vecs_dm, train_vecs_dbow))
print train_vecs_dm
print model_dm.most_similar("SENT_0")
#train over test set
#x_test = np.array(x_test)

for epoch in range(10):
#    perm = np.random.permutation(x_test.shape[0])
     model_dm.train(test_sentences)
#    model_dbow.train(x_test[perm])

#Construct vectors for test reviews
test_vecs_dm = getVecs(model_dm,test_sentences, size)
#test_vecs_dbow = getVecs(model_dbow, x_test, size)
print test_vecs_dm
#test_vecs = np.hstack((test_vecs_dm, test_vecs_dbow))

#Now use classifier on the train_vecs
