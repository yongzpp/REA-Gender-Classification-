from config import cfg
from sklearn.feature_extraction.text import TfidfVectorizer

class Preprocessor(object):
    '''
    Object to get the ngrams vector using TFIDF
    '''
    def __init__(self, name):
        '''
        Initalize TFIDF vectorizer
        :param name: series of human names to initialize vectorizer (pandas.Series)
        '''
        self.vectorizer = TfidfVectorizer(analyzer='char', \
                    ngram_range=(cfg.train.ngram[0],cfg.train.ngram[1])).fit(name)

    def vectorize(self, name):
        '''
        :param name: series of human names (pandas.Series)
        :return: word vectors using intialized vectorizer (sparse matrix: [n_samples, n_features])
        '''
        return self.vectorizer.transform(name)