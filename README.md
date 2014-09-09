py-arpa-lm
==========

Python API for reading and querying ARPA format language model.


Usage :

```
lm = LM("test.arpa")

# Score a sentence
lm.score("HELLO WORLD")

# Get integer word id
wId = lm.get_word_id("HELLO")

# Get the bigram probability of two words
lm.get_bg(lm.get_word_id("HELLO"),lm.get_word_id("WORLD"))

# Get the trigram probability of three words
lm.get_tg(lm.start,lm.get_word_id("HELLO"),lm.end)
```
