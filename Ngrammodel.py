from nltk.util import ngrams
from nltk import FreqDist
import nltk
import pandas as pd
import itertools
from nltk.corpus import stopwords
import pickle

class NgramModel:
   def __init__(self, unk, train, pad_left=False, pad_right=False):
      self.unk = unk
      self.pad_left = pad_left
      self.pad_right = pad_right
      self.words = [x for sent in train for x in sent]
      stop = set(stopwords.words('english'))
      self.unigram_without_stopwords = [x for x in self.words if x not in stop]
      self.unigram_fdist = FreqDist(self.unigram_without_stopwords)

      self.bigram = list(ngram for sent in train for ngram in ngrams(sent, 2, False, False))
      self.trigram = list(ngram for sent in train for ngram in ngrams(sent, 3, False, False))
      self.bigram_fdist = FreqDist(self.bigram)
      self.trigram_fdist = FreqDist(self.trigram)

      self.unigram_frequent = [unigram for unigram in self.unigram_fdist if self.unigram_fdist[unigram] > self.unk]
      self.bigram_frequent = [bigram for bigram in self.bigram_fdist if self.bigram_fdist[bigram] > self.unk]
      self.trigram_frequent = [trigram for trigram in self.trigram_fdist if self.trigram_fdist[trigram] > self.unk]

def isplit(iterable, splitters):
   return [list(g) for k, g in itertools.groupby(iterable, lambda x:x in splitters) if not k]

def main():
   csv_file = pd.read_csv('Kickstarter022.csv')
   sents = list()
   for i in range(len(csv_file['description'])):
      if i%1000 == 0:
         print(i, "th description working...")
      text = csv_file['description'][i]
      text = nltk.word_tokenize(text.lower())
      text = isplit(text, (".",";","!","?",))
      for sent in text:
         sents.append([x for x in list(filter(lambda a: a.isalnum(), sent))])
   del(csv_file)

   model = NgramModel(10, sents)

   file = open('ngramlist.txt', 'wb')

   full_list = [model.unigram_frequent, model.bigram_frequent, model.trigram_frequent]
   for i in range(3):
      myString = " ".join(map(str, full_list[i]))
      file.write(myString)
      file.write("\n")

   file.close()

if __name__ == "__main__":
   main()