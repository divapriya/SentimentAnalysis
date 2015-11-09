# -*- coding: utf-8 -*-
"""
Created on Mon Jun 01 17:35:00 2015

@author: PSarka
"""
# -*- coding: utf-8 -*-
"""
Created on Thu May 28 14:15:16 2015

@author: PSarka
"""
from bs4 import BeautifulSoup
import pandas as pd  
import csv
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
from nltk.corpus import wordnet
from nltk.tag import pos_tag
#from nltk.tag.stanford_POS import POSTagger
from nltk.tag.stanford import NERTagger
#import nltk.tokenizers.punkt
#from sklearn.feature_extraction.text import CountVectorizer
#from gensim.models import word2vec
import nltk.data
import re     

# Load dataset 
train = pd.read_csv("C:\Users\PSarka\Desktop\sentimentanalysis\labeledTrainData.tsv",header=0,delimiter="\t", quoting=3)
test = pd.read_csv(r"C:\Users\PSarka\Desktop\sentimentanalysis\testData.tsv",header=0,delimiter="\t", quoting=2)
sample = pd.read_csv("C:\Users\PSarka\Desktop\sentimentanalysis\sampleSubmission.csv")

#Proocessing Text
def remove_punctuations(text):
    wordlist=[]    
    sentences=sent_tokenize(text)
    for sentence in sentences:
        #words=word_tokenize(sentence)
        words=[word for word in sentence.split()]               
        words=[re.sub(r'[^\w\s]','',word) for word in words]     
        wordlist.append(words)
    return wordlist

def descriptive_words(words):
    meaningful_words=[]    
    tags=['VB','VBP','VBD','VBG','VBN','JJ','JJR','JJS','RB','RBR','RBS','UH']    
    #english_postagger = POSTagger(‘models/english-bidirectional-distsim.tagger’,‘stanford-postagger.jar’) 
    #english_postagger = POSTagger
    #for sentence in text:
    tagged_word=pos_tag(words)
    for word in tagged_word:            
        if word[1] in tags:
            meaningful_words.append(word[0])
    return meaningful_words        
                
def remove_names(text):
    meaningful_words=[]
    tags=['	Time', 'Location', 'Organization', 'Person', 'Money', 'Percent', 'Date']    
    #english_nertagger = NERTagger(‘C:\Python27\nltk_data\classifiers\stanford-ner-2014-08-27\classifiers\english.all.7class.distsim.crf.ser.gz’,‘C:\Python27\nltk_data\classifiers\stanford-ner-2014-08-27\stanford-ner.jar’)
    tagger = NERTagger.SocketNER(host='localhost', port=8080)
    tagger.get_entities("aleen")  
    english_nertagger = NERTagger
    for sentence in text:
        tagged_word=english_nertagger.tag(sentence.split())
    for word in tagged_word:
        if word[1] not in tags:
            meaningful_words.append(word[0])
    return meaningful_words
    
def negation_handling(wordlist):
    counter=False    
    wlist=[]    
    negations=["no","not","cant","cannot","never","less","without","barely","hardly","rarely","no longer","no more","noway","no where","by no means","at no time","not anymore","didnt"]
    for words in wordlist:       
        for i,j in enumerate(words):                           
            if j in negations and i<len(words)-1:             
                wlist.append(str(words[i]+'-'+words[i+1]))
                counter=True
            else:
                if counter is False:                
                    wlist.append(words[i])
                else:
                    counter=False
    return wlist
          
def synonyms(wordlist):
    wlist=[]    
    for word in wordlist:              
        print word        
        if wordnet.lemmas(word):
            string=str(wordnet.lemmas(word)[0])                                        
            
            wlist+=re.findall("Lemma\(\'(.+?)\.\w",string)
            print re.findall("Lemma\(\'(.+?)\.\w",string)
        else:
            wlist+=word
    return wlist


def processing(raw_review):
    word1=[]    
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    # 2. Remove Punctuations        
    letters_only = remove_punctuations(review_text) 
    # 3. Convert to lower case, split into individual words     
    for words in letters_only:
        wordset=[word.lower() for word in words]
        word1.append(wordset)                       
    #4Handling Double Negation
    negated_words=negation_handling(word1)
    #5 Read only verbs,adjectives,adverbs,interjections (descriptive words)  
    meaningful_words=descriptive_words(negated_words)           
    #6 Remove Time, Location, Organization, Person, Money, Percent, Date using NER   
    #removed_words=remove_names(meaningful_words)    
    #7. Remove stop words    
    stops =open(r'C:\Users\PSarka\Desktop\sentimentanalysis\stopwords.txt','r')   
    stops= set([word[:-1] for word in stops])  
    meaningful_words_new = [w for w in meaningful_words if not w in stops]    
    #6 Exchange similar words with there synonyms  
    syn_words=synonyms(meaningful_words_new)    
    #6.Stemming using Porter Stemmer,Lemming can also be used check which is more efficient
    st=PorterStemmer()
    stemmed_words=[st.stem(words) for words in syn_words]   
    #7. Join the words back into one string separated by space, 
    # and return the result.    
    return( " ".join(stemmed_words ))   

print "Cleaning train started"
clean_train_reviews=[]
#processing(" \\I did'nt like this convincing movie// didnt. Although it could had been better")
#print train["review"][0]
for i in xrange( 0,1):#train["review"].size): #Remember to change 5 with train["review"].size 
    clean_train_reviews.append([processing( train["review"][i] ),train["sentiment"][i]] ) 
print "Cleaning Test Completed"

