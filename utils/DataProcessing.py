from nltk import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, SnowballStemmer


class DataProcessing():

    def getTokenizedWords(self, data):
        return word_tokenize(data)

    def getTokenizedSentences(self, data):
        sentences = sent_tokenize(data)
        return sentences

    def removeStopWords(self, data):
        stopwordsList = stopwords.words("english")
        filteredwords = []
        for lineWord in data:
            filteredwords.append(list(filter(lambda word: (word.lower() not in stopwordsList) and len(word) > 2, lineWord)))
        return filteredwords

    def getStemWords(self, data, portStemmer = True):
        stemmedWords = []
        if portStemmer:
            portStem = PorterStemmer()
            for words in data:
                stemmedWords.append([portStem.stem(word) for word in words])
        else:
            snowballStem = SnowballStemmer()
            for words in data:
                stemmedWords.append([snowballStem.stem(word) for word in words])

        return stemmedWords
