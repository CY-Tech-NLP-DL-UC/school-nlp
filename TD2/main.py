import numpy as np
import pandas as pd
import re
import string
import nltk
from nltk.corpus import stopwords as stpw
#nltk.download('stopwords')
#nltk.download('punkt')
from nltk.tokenize import word_tokenize
import time

def preprocess(content):
    content = content.lower()
    content = content.translate(str.maketrans("","", string.punctuation))
    content = re.sub(r'\d+ *|\b[a-z]\b *', "", content) # remove isolated letters
    content = content.strip()
    tokens = word_tokenize(content)
    stopwords = set(stpw.words("english"))
    # removes stopwords and duplicates
    content = " ".join(
        list(dict.fromkeys([t for t in tokens if not t in stopwords]))
    )
    return content


class NaiveBayes:
    """
    Classificator based on Naive Bayes' method to make predictions on tweets (Spam filtering)
    Parameters
    ----------
    knowledge: Pandas DataFrame
        DataFrame of the word occurences for each label/class.
    labels: Pandas Series
        Serie of labels.
    """
    def __init__(self, knowledge, labels):
        self.knowledge = knowledge
        self.labels = labels
        self.alpha = 1
        self.N = len(self.knowledge.words)
        self.priors = self.computes_priors(self.labels)

    def computes_priors(self, labels):
        """ Computes prior probabilities.
        Parameter
        ---------
        labels: Pandas Series
            Serie of labels.
        Recall
        ------
        prior is P(label=l_i).
        """
        priors = []
        for count in labels.value_counts():
            priors.append(count/len(labels))            

        return priors

    def computes_likelihood(self, word, label):
        """ Computes likelihood of the existence of a word in a sentence
        knowing a the sentence's label.
        Parameters
        ----------
        word: string

        label: string
        """
        occ = self.knowledge.loc[self.knowledge.words == word, label].item()
        total = self.labels.value_counts()[label]
        
        return (occ + self.alpha) / (total + self.alpha * self.N)

    def predict(self, sentence):
        """ Predicts a label for a sentence.
        Parameter
        ---------
        sentence: string list.
        """
        ham_p = self.priors[0]
        spam_p = self.priors[1]

        for word in sentence:
            if word in list(self.knowledge.words):
                ham_p *= self.computes_likelihood(word, "ham")
                spam_p *= self.computes_likelihood(word, "spam")
        
        if ham_p > spam_p:
            return "ham"
        elif spam_p > ham_p:
            return "spam"
        else:
            return "unknown"

    def accuracy(self, predicted, expected):
        """ Computes accuracy according to both lists of
        expected and predicted values.
        Parameters
        ----------
        predicted: list

        expected: list
        """
        data = pd.DataFrame({
            'expected': expected,
            'predicted': predicted
        })
        return len(data[data.expected == data.predicted])/len(data)

    def precision(self, predicted, expected, label):
        """ Computes the precision on a label according to
        both lists of expected and predicted values.
        Parameters
        ----------
        predicted: list

        expected: list

        label: string
        """
        data = pd.DataFrame({
            'expected': expected,
            'predicted': predicted
        })
        data = data[data.expected == label]
        return len(data[data.expected == data.predicted])/len(data)

    def recall(self, predicted, expected, label):
        """ Computes recall on a label according to both lists
        of expected and predicted values.
        Parameters
        ----------
        predicted: list

        expected: list

        label: string
        """
        data = pd.DataFrame({
            'expected': expected,
            'predicted': predicted
        })
        data = data[data.predicted == label]
        return len(data[data.expected == data.predicted])/len(data)

    def f1_score(self, predicted, expected, label):
        """ Computes F1 score on a label according to both lists of
        expected and predicted values.
        Parameters
        ----------
        predicted: list

        expected: list

        label: string
        """
        precision_ = self.precision(predicted, expected, label)
        recall_ = self.recall(predicted, expected, label)
        return 2 * (precision_*recall_) / (precision_ + recall_)

    def eval(self, predicted, expected):
        """
        Prints infos on classificator's efficiency based on expectations and results
        """
        print(f"Acccuracy: {self.accuracy(predicted, expected)}")
        print(f"Precision on ham label: {self.precision(predicted, expected, 'ham')}")
        print(f"Precision on spam label: {self.precision(predicted, expected, 'spam')}")
        print(f"Recall on ham label: {self.recall(predicted, expected, 'ham')}")
        print(f"Recall on spam label: {self.recall(predicted, expected, 'spam')}")
        print(f"F1 score on ham label: {self.f1_score(predicted, expected, 'ham')}")
        print(f"F1 score on spam label: {self.f1_score(predicted, expected, 'spam')}")


if __name__ == "__main__":
    start = time.time()

    data = pd.read_csv("SMS_source.csv")

    np.random.seed(0) # to get the same randomness
    threshold = np.random.rand(len(data)) < 0.7
    train_set = data[threshold]
    test_set = data[~threshold]
    print(f"{len(train_set)} training instances (~70%).\n{len(test_set)} testing instances (~30%).")

    ham_prop = len(train_set[train_set.labels == "ham"])/len(train_set) * 100
    spam_prop = len(train_set[train_set.labels == "spam"])/len(train_set) * 100

    print(f"ham: {ham_prop:.4}%\nspam: {spam_prop:.4}%")

    ham_prop = len(test_set[test_set.labels == "ham"])/len(test_set) * 100
    spam_prop = len(test_set[test_set.labels == "spam"])/len(test_set) * 100

    print(f"ham: {ham_prop:.4}%\nspam: {spam_prop:.4}%")

    train_set.contents = train_set.contents.apply(preprocess)

    spam_data = train_set[train_set.labels == "spam"]
    spam_freq = len(spam_data)/len(train_set)
    ham_data = train_set[train_set.labels == "ham"]
    ham_freq = len(ham_data)/len(train_set)
    print(f"Number of spam instances: {len(spam_data)}.\nNumber of ham instances: {len(ham_data)}.")

    # Spam
    spam_contents = (" ".join(spam_data.contents)).split(" ")
    uniq_words = list(dict.fromkeys(spam_contents))
    occurence_spam_data = pd.DataFrame(
        {
            'word': uniq_words,
            'occurence': [ spam_contents.count(word) for word in uniq_words],
        }
    )
    occurence_spam_data = occurence_spam_data.sort_values(by="occurence", ascending=False)

    # Ham
    ham_contents = (" ".join(ham_data.contents)).split(" ")
    uniq_words = list(dict.fromkeys(ham_contents))
    occurence_ham_data = pd.DataFrame(
        {
            'word': uniq_words,
            'occurence': [ ham_contents.count(word) for word in uniq_words],
        }
    )
    occurence_ham_data = occurence_ham_data.sort_values(by="occurence", ascending=False)

    uniq_words = list(dict.fromkeys(list(occurence_ham_data.word)+list(occurence_spam_data.word)))
    print(f"Total number of words: {len(uniq_words)}.")

    knowledge = pd.DataFrame(
        {
            "words": uniq_words,
            "ham": [
                occurence_ham_data[occurence_ham_data.word == word].occurence.item()
                if len(occurence_ham_data[occurence_ham_data.word == word]) > 0 else 0
                for word in uniq_words
            ],
            "spam": [
                occurence_spam_data[occurence_spam_data.word == word].occurence.item()
                if len(occurence_spam_data[occurence_spam_data.word == word]) > 0 else 0
                for word in uniq_words
            ]
        }
    )

    test_set.contents = test_set.contents.apply(preprocess)

    clf = NaiveBayes(knowledge, train_set.labels)
    expected = list(test_set.labels)
    predicted = [clf.predict(sentence.split(" ")) for sentence in test_set.contents]
    clf.eval(predicted, expected)

    print("Execution time : " + str(time.time() - start))