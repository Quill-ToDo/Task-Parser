from microtc.utils import tweet_iterator
from os.path import join
import spacy


nlp = spacy.load("en_core_web_sm")
doc = nlp("Ana is going to New York next Friday!")

for x in doc:
    print(x.text, x.pos_)
