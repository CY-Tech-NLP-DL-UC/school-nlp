from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer

import pandas as pd

import time

import nltk
#nltk.download('averaged_perceptron_tagger')
from nltk.corpus import stopwords as stpw
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import string 
import re 
#nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer

def eval(test_label, predicted):
    """
    Computes confusion matrix and outputs the classifier's statistic
    Parameters
    ----------
    test_label : array-like of shape (n_samples,)

    predicted : array-like of shape (n_samples,)
    """
    tn, fp, fn, tp = confusion_matrix(test_label, predicted).ravel() # Binary case

    print(f"tn : {tn}, fp : {fp}, fn : {fn}, tp : {tp}")
    log_stats(tn, fp, fn, tp)

def log_stats(tn, fp, fn, tp):
    """
    Computes accuracy, precision, recall, f1 score
    Parameters
    ----------
    tn : int

    fp : int

    fn : int

    tp : int
    """
    # Accuracy

    acc = (tn+tp)/(tn+fp+fn+tp)
    pre = tp/(tp+fp)
    recall = tp/(tp+fn)
    f1_score = 2*(recall * pre) / (recall + pre)
    
    # Log

    print(f"Accuracy : {acc}")
    print(f"Precision : {pre}")
    print(f"Recall : {recall}")
    print(f"F1 Score : {f1_score}")

def preprocess_v1(content) : 

    # remove upper letters
    content = content.lower()
    
    # remove punctuation 
    content = content.translate(str.maketrans("","", string.punctuation))
    
    # remove isolated letters
    content = re.sub(r'\d+ *|\b[a-z]\b *', "", content) 
    content = content.strip()
    
    tokens = word_tokenize(content)

    # lemmatization 
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(word) for word in tokens]
    
    # clean stop words 
    stopwords = set(stpw.words("english")) 
    stopwords = stopwords.union(['movie'])
 
    # removes stopwords and duplicates
    content = " ".join(
        list(dict.fromkeys([t for t in tokens if not t in stopwords]))
    )  
    
    return content

def preprocess_v2(content) : 
    # remove proper nouns 
    content = nltk.tag.pos_tag(content.split())
    content = [word for word,tag in content if tag != 'NNP' and tag != 'NNPS' and tag != 'VB' and tag != 'VBD' and tag != 'VBG' and tag != 'VBN' and tag != 'VBP' and tag != 'VBZ']
    content = " ".join(content)
    
    # remove upper letters
    content = content.lower()
    
    # remove punctuation 
    content = content.translate(str.maketrans("","", string.punctuation))
    
    # remove isolated letters
    content = re.sub(r'\d+ *|\b[a-z]\b *', "", content) 
    content = content.strip()
    
    tokens = word_tokenize(content)

    # clean stop words 
    stopwords = set(stpw.words("english")) 
    stopwords = stopwords.union(['movie'])
 
    # removes stopwords and duplicates
    content = " ".join(
        list(dict.fromkeys([t for t in tokens if not t in stopwords]))
    )  
    
    return content 

if __name__ == '__main__':
    """
    train_set1 = pd.read_csv("train_dataset.csv")
    train_set2 = train_set1.copy(deep=True)
    test_set1 = pd.read_csv("test_dataset.csv")
    test_set2 = test_set1.copy(deep=True)

    # use preprocessor attribute to preprocess the data, and max_features to set a boundary on the maximum number of words used to create the bag of words.
    m1 = CountVectorizer(max_features=1000, preprocessor=preprocess_v1)
    m2 = CountVectorizer(max_features=1000, preprocessor=preprocess_v2)

    # learns the vocabulary dictionnary and returns document-term matrix as a Numpy Array.
    X_train1 = m1.fit_transform(train_set1.features).toarray()
    X_train2 = m2.fit_transform(train_set2.features).toarray()
    
    # transforms data to data-term matrix as a Numpy Array.
    X_test1 = m1.transform(test_set1.features).toarray()
    X_test2 = m2.transform(test_set2.features).toarray()
    
    # gets train set labels.
    y_train1 = train_set1.label.to_numpy()
    y_train2 = train_set2.label.to_numpy()
    
    # gets test set labels.
    y_test1 = test_set1.label.to_numpy()
    y_test2 = test_set2.label.to_numpy()

    # Random Forest classifier
    clf1 = RandomForestClassifier().fit(X_train1, y_train1)
    clf2 = RandomForestClassifier().fit(X_train2, y_train2)

    predicted1 = clf1.predict(X_test1)
    predicted2 = clf2.predict(X_test2)

    # Evaluate

    eval(y_test1, predicted1)
    eval(y_test2, predicted2)
    """
    print(preprocess_v1("I have to t2ell you I've been a FAN of S%tar Trek TNG since i was a kid.<br />"))
    print(preprocess_v2("I have to t2ell you I've been a FAN of S%tar Trek TNG since i was a kid.<br />"))