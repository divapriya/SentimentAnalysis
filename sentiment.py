from bs4 import BeautifulSoup
import pandas as pd  
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
#from sklearn.feature_extraction.text import CountVectorizer
#from gensim.models import word2vec
import nltk.data
import re     

# Load dataset 
train = pd.read_csv("labeledTrainData.tsv",header=0,delimiter="\t", quoting=3)
test = pd.read_csv("testData.tsv",header=0,delimiter="\t", quoting=2)
sample = pd.read_csv("sampleSubmission.csv")

#Proocessing Text
def processing(raw_review):
    # Function to convert a each raw review to a string of words
    # 1. Remove HTML
    review_text = BeautifulSoup(raw_review).get_text() 
    #
    # 2. Remove non-letters        
    letters_only = re.sub("[^a-zA-Z]", " ", review_text) 
    #
    # 3. Convert to lower case, split into individual words
    words = letters_only.lower().split()                             
    #
    # 4. In Python, searching a set is much faster than searching
    #   a list, so convert the stop words to a set
    stops = set(stopwords.words("english"))                  
    # 
    # 5. Remove stop words
    meaningful_words = [w for w in words if not w in stops]   
    #
    #6.Stemming using Porter Stemmer,Lemming can also be used check which is more efficient
    st=PorterStemmer()
    stemmed_words=[st.stem(words) for words in meaningful_words]
    #7. Join the words back into one string separated by space, 
    # and return the result.
    return( " ".join(stemmed_words ))   

clean_train_reviews=[]
for i in xrange( 0,5): #Remember to change 5 with train["review"].size 
    clean_train_reviews.append([processing( train["review"][i] ),train["sentiment"][i]] )


def get_words_in_tweets(reviews):
    all_words = []
    for (words,sentiment) in reviews:
            all_words.append(words)
    return all_words

def get_word_features(wordlist):
    wordlist = nltk.FreqDist(wordlist)
    word_features = wordlist.keys()
    return word_features

word_features = get_word_features(get_words_in_tweets(clean_train_reviews)) #Frequency distribution of trained data

def extract_features(document): #Checks wether testing data is present in training data
    document_words = set(document)
    features = {}
    for word in word_features:
       features['%s' % word] = (word in document_words)
    return features


training_set = nltk.classify.apply_features(extract_features,clean_train_reviews)
classifier = nltk.NaiveBayesClassifier.train(training_set)

tweet = 'Larry is my friend'
print classifier.classify(extract_features(tweet.split()))