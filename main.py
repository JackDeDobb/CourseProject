##Use Python3

iimport json
import os
import nltk
import string
import numpy as np
stemmer = nltk.stem.porter.PorterStemmer()
try:
     _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
     pass
 else:
     ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')



def parseWords():
  # Use nltk and stopwords to tokenize words
  tokenizedWords=[]
  sentences = nltk.sent_tokenize(content)
  for sent in sentences:
      words = nltk.word_tokenize(sent)
      stemmedWords = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
      stemmedWordsWithoutStopwords = [v for v in temp if v not in stopWords] # Removve stopwords
      if len(stemmedWordsWithoutStopwords)>0:
          tokenizedWords.append(stemmedWordsWithoutStopwords)
  return tokenizedWords
  #pass

def genStopwords():
  # Create stopwords
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))

def readData():
  # Load and read all json files
  pass

def createVocab():
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allTerms = []
  for reviewData in reviewDataLists:
      for review in reviewData["Reviews"]:
          parseWordsInReview= []
          for parseWord in parseWords(review['Content'],stopwords):
                  parseWordsInReview = parseWord + parseWordsInReview
          allTerms +=  parseWordsInReview
  termFrequency = nltk.FreqDist(allTerms)
  vocab = []
  cnt = []
  vocabDict={}
  for k,v in termFrequency.items():
      if v>5:
          vocab.append(k)
          cnt.append(v)
  vocab = np.array(vocab)[np.argsort(vocab)].tolist()
  cnt = np.array(cnt)[np.argsort(vocab)].tolist()
  vocabDict = dict(zip(vocab,range(len(vocab))))
  return vocab, cnt, vocabDict
  #pass # return vocabDict

def saveFile(path):
  # Save to file
  pass

def loadFile(path):
  # Load from file
  pass

def generateAspectTerms(aspectLines,vocabDict):
  # Go through each line of the fule and use vocabdict to get the tokenized word
  pass


def addAspectWords(analyzer, p, NumIter,c):
  # Update aspect Words based on Expectation step of EM algorithm
  pass



def getVocab():
  ##### Step 1: Create vocabulary from json files
  stopWords= getStopwords('StopWords.json')  ### generate a list of stopwords
  reviewDataList=getData('HotelData/CleanData') ##Read the json files
  return createVocab(reviewDataList,stopWords)


def runBootstrap():
  ##### step 2. Run bootstrapping method on vocab
  # Loading vocab data from saved file
  vocab, cnt, vocabDict =getVocab()

  # Load aspect words
  aspectLines=loadFile(aspectWordsFilePath)
  # Get the stemmed term for each aspect from the vocabDict
  aspectTerms=generateAspectTerms(aspectLines,vocabDict)

  #### Run EM algorithm on aspect keywords and save it to a file
  addAspectWords(vocabDict)

  # Create the word matrix for all the reviews
  createWordMatrix()
  # Use the word matrix to generate the results
  generateResults()
