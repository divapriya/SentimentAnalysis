from bs4 import BeautifulSoup
import pandas as pd  
import csv
from nltk.stem.porter import PorterStemmer
from nltk.tokenize import word_tokenize,sent_tokenize
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
    #english_nertagger = NERTagger(‘classifiers/english.all.7class.distsim.crf.ser.gz’,‘stanford-ner.jar’)
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
    negations=["no","not","cant","never","less","without","barely","hardly","rarely","no longer","no more","noway","no where","by no means","at no time","not anymore","didnt"]
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
    #6.Stemming using Porter Stemmer,Lemming can also be used check which is more efficient
    st=PorterStemmer()
    stemmed_words=[st.stem(words) for words in meaningful_words_new]   
    #7. Join the words back into one string separated by space, 
    # and return the result.
    print stemmed_words    
    return( " ".join(stemmed_words ))   


print "Cleaning train started"
clean_train_reviews=[]
for i in xrange( 0,4000):#train["review"].size): #Remember to change 5 with train["review"].size 
    clean_train_reviews.append([processing( train["review"][i] ),train["sentiment"][i]] ) 
print "Cleaning Test Completed"


#resultFyle = open("processedtext.csv",'wb')
#wr = csv.writer(resultFyle, dialect='excel')
#for item in clean_train_reviews:
#   wr.writerow(item)


#Creating Dictionary of the training words
def get_words_in_tweets(reviews):
    all_words = []
    for (words,sentiment) in reviews:
        word=words.split()       
        for w in word:
            all_words.append(w)
    return all_words

print "Dictionary of words found in trainig data set"

#Creating Frequency Distribution of the training words
def get_word_features(wordlist):     
    all_words=[]    
    word_features=[]
    for word in wordlist:
            all_words.append(word)
    #print all_words
    wordlist = nltk.FreqDist(all_words)        
    wordlist.plot()    
    wordlist=wordlist.items()
    for items,freq in wordlist:
        if freq>100:
            word_features.append(items)
    #print word_features
    print len(word_features)
    return word_features

print"Display of frequency distribution of words"
word_features = get_word_features(get_words_in_tweets(clean_train_reviews)) #Frequency distribution of trained data


#Checks wether testing data is present in training data and creates the table
def extract_features(document): 
    features = {}
    for word2 in word_features:
        features['%s' % word2]=0
    document=document.split()
    for word1 in document:    
        for word2 in word_features:
            if word1 in word2:           
                features['%s' % word2] += 1
    return features

#print"printing the table"
#features=[]
#for i in xrange(0,10):  
#    features.append(extract_features(processing(test["review"][i])))
#    for items in features:      
#        print items      
  
#Training the Classifier
print "Classifier training started"
training_set = nltk.classify.apply_features(extract_features,clean_train_reviews)
print training_set
classifier = nltk.NaiveBayesClassifier.train(training_set)
print "Classifier training over"

#Testing the Classifier on the test data
#tweet="i hated this movie, its too bad"
#print classifier.classify(extract_features(tweet))
#Output Final Sentiment Predictions
test_reviews=[]#contains sentiments of the sentences
#for i in xrange(0,5):#Remember to change 5 with test["review"].size 
#    tweet=processing(test["review"][i])
#    print test["review"][i],classifier.classify(extract_features(tweet))
#   test_reviews.append([test["review"][i],classifier.classify(extract_features(tweet))])
#resultFyle = open("resultsentiments.csv",'wb')
#wr = csv.writer(resultFyle, dialect='excel')
#for item in test_reviews:
#    wr.writerow(item)    
   
#Calculating Accuracy 
accuracy_reviews=[]
for i in xrange(4500,5500):  
   accuracy_reviews.append([extract_features(processing(train["review"][i])),train["sentiment"][i]])

print 'Accuracy:',nltk.classify.accuracy(classifier,accuracy_reviews)    
