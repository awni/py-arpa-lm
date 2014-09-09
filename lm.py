import collections
import math

NGRAM_T = '\\%d-grams'
UNK = "<unk>"
START = "<s>"
END = "</s>"

class LM:
    def __init__(self,arpafile=None,start=START,end=END,unk=UNK,fromFile=None):
        """
        Load arpafile to get words and assign ids
        Unigram table indexed by word id into tuple of prob and backoff
        Bigram table indexed by (word1id, word2id) -> prob
        """
        if fromFile is not None:
            self.from_file(fromFile)
            return 

        fid = open(arpafile,'r')
        self.read_header(fid)
        self.wordToInt = dict()
        self.unigrams = dict()
        self.bigrams = dict()
        self.scale = math.log(10) #scale everything from log10 to ln
        if self.isTrigram:
            self.trigrams = dict()
            self.bigrams = dict()

        self.dict_to_default_dict()

        self.load_ug(fid) # read unigram lm and word map, TODO provide list of allowed chars and scrub those not contained
        self.load_bg(fid) # read bigram lm
        if self.isTrigram: # if trigram lm, read that too
            self.load_tg(fid)
        self.start = self.wordToInt[start]
        self.end = self.wordToInt[end]
        self.unk = self.wordToInt[unk]
        fid.close()

    def dict_to_default_dict(self):
        self.wordToInt = collections.defaultdict(lambda : -1,self.wordToInt)
        self.unigrams = collections.defaultdict(lambda : (0.0,0.0),self.unigrams)
        if self.isTrigram:
            self.trigrams = collections.defaultdict(int,self.trigrams)
            self.bigrams = collections.defaultdict(lambda : (0.0,0.0),self.bigrams)
        else:
            self.bigrams = collections.defaultdict(int,self.bigrams)

    def default_dict_to_dict(self):
        self.wordToInt = dict(self.wordToInt)
        self.unigrams = dict(self.unigrams)
        if self.isTrigram:
            self.trigrams = dict(self.trigrams)
            self.bigrams = dict(self.bigrams)
        else:
            self.bigrams = dict(self.bigrams)


    def read_header(self,fid):
        while fid.readline().strip() != "\\data\\":
            continue
        line = fid.readline()
        assert'ngram 1' in line, "Something wrong with file format."
        self.numWords = int(line.strip().split('=')[1])
        line = fid.readline()
        assert 'ngram 2' in line, "Must at least provide bigram LM"
        line = fid.readline()
        if 'ngram 3' in line:
            self.isTrigram = True
        else:
            self.isTrigram = False

    def load_ug(self,fid):
        count = 0
        while NGRAM_T%1 not in fid.readline():
            continue
        while True:
            line = fid.readline().strip().split()
            if len(line)==0:
                break
            self.wordToInt[line[1]] = count
            if len(line)==3:
                self.unigrams[count] = (self.scale*float(line[0]),
                                        self.scale*float(line[2]))
            else:
                self.unigrams[count] = (self.scale*float(line[0]),0.0)
            count += 1
 
    def load_bg(self,fid):
        while NGRAM_T%2 not in fid.readline():
            continue
        while True:
            line = fid.readline().strip().split()
            if len(line)==0 or "\\end\\" == line[0]:
                break
            key = (self.wordToInt[line[1]],self.wordToInt[line[2]])
            if self.isTrigram: 
                if len(line)==4:
                    self.bigrams[key] = (self.scale*float(line[0]),
                                         self.scale*float(line[3]))
                else:
                    self.bigrams[key] = (self.scale*float(line[0]),
                                         0.0)
            else:
                self.bigrams[key] = self.scale*float(line[0])

    def load_tg(self,fid):
        while NGRAM_T%3 not in fid.readline():
            continue
        while True:
            line = fid.readline().strip().split()
            if len(line)==0 or "\\end\\" == line[0]:
                break
            key = (self.wordToInt[line[1]],self.wordToInt[line[2]],
                   self.wordToInt[line[3]])
            self.trigrams[key] = self.scale*float(line[0])

    def get_word_id(self,word):
        """
        Returns word id for words in vocab and UNK id otherwise.
        """
        id = self.wordToInt[word]
        if id==-1:
            return self.unk
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

    def tg_prob(self,w1,w2,w3):
        """
        Returns trigram probability of p(w3 | w1, w2) :

        p(wd3|wd1,wd2)= if(trigram exists)           p_3(wd1,wd2,wd3)
                        else if(bigram w1,w2 exists) bo_wt_2(w1,w2)*p(wd3|wd2)
                        else                         p(wd3|w2)

        p(wd2|wd1)= if(bigram exists) p_2(wd1,wd2)
                    else              bo_wt_1(wd1)*p_1(wd2)
        """
        val = self.trigrams[(w1,w2,w3)]
        # backoff to bigram
        if val == 0:
            val += self.bigrams[(w2,w3)][0]
            # backoff to unigram
            if val == 0:
                val += self.unigrams[w3][0]
                val += self.unigrams[w2][1]
            val += self.bigrams[(w1,w2)][1]
        return val

    def score_bg(self,sentence):
        words = [self.get_word_id(w) for w in sentence.strip().split()]
        val = 0.0
        val += self.bg_prob(self.start,words[0])
        for i in range(len(words)-1):
            val += self.bg_prob(words[i],words[i+1])
        #val += self.bg_prob(words[-1],self.end)
        return val

    def score_tg(self,sentence):
        assert self.isTrigram,\
                "Can't score sentence as trigram with bigram lm."
        words = [self.get_word_id(w) for w in sentence.strip().split()]
        val = 0.0
        if len(words) == 1:
            w1 = self.start
            w3 = self.end
        else:
            w1 = words[-2]
            w3 = words[1]
        val += self.tg_prob(self.start,self.start,words[0])
        val += self.tg_prob(self.start,words[0],w3)
        for i in range(len(words)-2):
            val += self.tg_prob(words[i],words[i+1],words[i+2])
        #if w3 != self.end:
        #    val += self.tg_prob(w1,words[-1],self.end)
        return val

    def to_file(self,file):
        import cPickle as pickle
        self.default_dict_to_dict()
        with open(file,'w') as fid:
            pickle.dump([self.isTrigram, self.start, self.end, self.unk, self.scale],fid)
            pickle.dump(self.wordToInt,fid)
            pickle.dump(self.unigrams,fid)
            pickle.dump(self.bigrams,fid)
            if self.isTrigram:
                pickle.dump(self.trigrams,fid)

    def from_file(self,file):
        import cPickle as pickle
        with open(file,'r') as fid:
            self.isTrigram, self.start, self.end, self.unk, self.scale = pickle.load(fid)
            self.wordToInt = pickle.load(fid)
            self.unigrams = pickle.load(fid)
            self.bigrams = pickle.load(fid)
            if self.isTrigram:
                self.trigrams = pickle.load(fid)
        self.dict_to_default_dict()

if __name__=='__main__':
#    lm = LM('/afs/cs.stanford.edu/u/awni/wsj/ctc-utils/lm_bg.arpa')
#    print lm.wordToInt['ACCEPT']
#    print lm.score_bg("HELLO AGAIN")
#    print lm.score_bg("HELLO ABDO")

#    lm = LM('/afs/cs.stanford.edu/u/awni/wsj/ctc-utils/lm_tg.arpa')
    lm = LM('/afs/cs.stanford.edu/u/awni/wsj/data/local/nist_lm/test.arpa')

