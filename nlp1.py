'''
	pluralsight intro to natural language processing course
	
'''

import nltk

text = "Mary had a little lamb. Her fleece was white as snow"
from nltk.tokenize import word_tokenize, sent_tokenize
sents = sent_tokenize(text)
print(sents)
words = word_tokenize(text)
print(words)

from nltk.corpus import stopwords
from string import punctuation
custom_stop_words = set(stopwords.words('english') + list(punctuation))

words_filtered = list(word for word in word_tokenize(text) if word not in custom_stop_words)
print(words_filtered)

#collocations
from nltk.collocations import *
bigram_measures = nltk.collocations.BigramAssocMeasures()
finder = BigramCollocationFinder.from_words(words_filtered)
print(finder.ngram_fd.items())

#stemming and tagging parts of speech
from nltk.stem.lancaster import LancasterStemmer
st = LancasterStemmer()
stemmed_words = [st.stem(word) for word in word_tokenize(text)]
print(stemmed_words)

print(nltk.pos_tag(word_tokenize(text)))
'''
	wordnet: an nltk corpus reader
	
'''
from nltk.corpus import wordnet as wn
for ss in wn.synsets('bass'):
	print(ss, ss.definition()); #words, definitions, parts of speech
	
	
#lesk algorithm: for word sense disambiguation
from nltk.wsd import lesk
sense1 = lesk(word_tokenize("Sing in a lower tone, along with the bass"),"bass")
print(sense1)