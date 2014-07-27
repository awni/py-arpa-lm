import collections
import math

NGRAM_T = '\\%d-grams'
UNK = "<UNK>"
START = "<s>"
END = "</s>"

class LM:
    def __init__(self,arpafile,ngram=2):
        """
        Load arpafile to get words and assign ids
        Unigram table indexed by word id into tuple of prob and backoff
        Bigram table indexed by (word1id, word2id) -> prob
        """
        self.numWords = 0
        self.wordToInt = collections.defaultdict(lambda : -1)
        self.unigrams = collections.defaultdict(int)
        self.bigrams = collections.defaultdict(int)
        scale = math.log(10) #scale everything from log10 to ln
        count = 0
        with open(arpafile,'r') as fid:
            while fid.readline().strip() != "\\data\\":
                continue
            self.numWords = int(fid.readline().strip().split('=')[1])
            while NGRAM_T%1 not in fid.readline():
                continue
            while True:
                line = fid.readline().strip().split()
                if len(line)==0:
                    break
                self.wordToInt[line[1]] = count
                self.unigrams[count] = (scale*float(line[0]),
                                        scale*float(line[2]))
                count += 1
            while NGRAM_T%2 not in fid.readline():
                continue
            while True:
                line = fid.readline().strip().split()
                if len(line)==0 or "\\end\\" == line[0]:
                    break
                key = (self.wordToInt[line[1]],self.wordToInt[line[2]])
                self.bigrams[key] = scale*float(line[0])

    def get_word_id(self,word):
        """
        Returns word id for words in vocab and UNK id otherwise.
        """
        id = self.wordToInt[word]
        if id=="-1":
            return self.wordToInt[UNK]
        else:
            return id

    def ug_prob(self,wid):
        """
        Returns unigram probility of word.
        """
        return self.unigrams[wid][0]

    def bg_prob(self,w1,w2):
        """
        Returns bigram probability p(w2 | w1),
        uses backoff if bigram does not exist.
        """
        key = (w1,w2)
        val = self.bigrams[key]
        if val == 0:
            val += self.unigrams[w1][1] + self.unigrams[w2][0]
        return val

    def score(self,sentence):
        words = sentence.strip().split()
        val = 0.0
        val += self.bg_prob(self.get_word_id(START),
                    self.get_word_id(words[0]))
        for i in range(len(words)-1):
            val += self.bg_prob(self.get_word_id(words[i]),
                        self.get_word_id(words[i+1]))
        val += self.bg_prob(self.get_word_id(words[-1]),
                    self.get_word_id(END))
        return val

if __name__=='__main__':
    lm = LM('lm_bg.arpa')
    print lm.score("HELLO AGAIN")

