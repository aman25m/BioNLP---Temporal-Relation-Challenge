import nltk

class posTagging():

    def __init__(self):
        self.pos_wordList = []


    def posTagging(self, data):
        for wordlist in data:
            self.pos_wordList.append(nltk.pos_tag(wordlist))
        return self.pos_wordList