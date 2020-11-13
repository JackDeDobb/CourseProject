# Use Python3
import json
import os
import nltk
import string
import numpy as np


stemmer = nltk.stem.porter.PorterStemmer()
try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
# The try block does not raise any errors, so the else block is executed
else:
     ssl._create_default_https_context = _create_unverified_https_context
nltk.download('punkt')


def parseWordsForSentence(content, vocab, vocabDict):
  # Use nltk and stopwords to tokenize words for  each review
  tokenizedWords = []
  sentences = nltk.sent_tokenize(content)
  for sent in sentences:
    words = nltk.word_tokenize(sent)
    stemmedWordsInVocab = [stemmer.stem(w.lower()) for w in words if stemmer.stem(w.lower()) in vocabDict]
    words = [vocabDict.get(w) for w in stemmedWordsInVocab]
    if len(words) > 0:
      tokenizedWords.append(words)
  return tokenizedWords


def parseWords(content, stopWords):
  # Use nltk and stopwords to tokenize words
  tokenizedWords = []
  sentences = nltk.sent_tokenize(content)
  for sent in sentences:
    words = nltk.word_tokenize(sent)
    stemmedWords = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
    stemmedWordsWithoutStopwords = [v for v in stemmedWords if v not in stopWords] # Remove stopwords
    if len(stemmedWordsWithoutStopwords) > 0:
      tokenizedWords.append(stemmedWordsWithoutStopwords)
  return tokenizedWords

def genStopwords():
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))

def createVocab(reviewDataList,stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allTerms = []
  for reviewData in reviewDataList:
    for review in reviewData['Reviews']:
      parseWordsInReview = []
      for parseWord in parseWords(review['Content'], stopWords):
        parseWordsInReview = parseWord + parseWordsInReview
      allTerms += parseWordsInReview
  termFrequency = nltk.FreqDist(allTerms)
  vocab = []
  cnt = []
  vocabDict = {}
  for k,v in termFrequency.items():
    if v > 5:
      vocab.append(k)
      cnt.append(v)
  vocab = np.array(vocab)[np.argsort(vocab)].tolist()
  cnt = np.array(cnt)[np.argsort(vocab)].tolist()
  vocabDict = dict(zip(vocab, range(len(vocab))))
  return vocab, cnt, vocabDict

def generateAspectTerms(aspectLines,vocabDict):
  # TODO: Aspect modeling
  pass

def addAspectWords():
  # TODO: Update aspect Words based on Expectation step of EM algorithm
  pass

def calculateSForReview(reviewWords, aspectTerms, vocab):
  # Calculate s for each review and aspect
  s_aspect_word = np.zeros(len(aspectTerms),len(reviewWords))
  for aspect in range(len(aspectTerms)):
    for word in reviewWords:
      i = vocab.index(word) # Get the index
      s_aspect_word[aspect,i] = num_aspect_word[aspect,i]+1
  return s_aspect_word

def createWMatrixForEachReview(review, vocab, vocabDict, aspectTerms):
  # Generate the matrix for each review
  reviewWords = parseWordsForSentence(review, vocab, vocabDict)
  reviewMatrix = np.zeros((len(aspectTerms),len(reviewWords)))
  s_aspect_word = calculateSForReview(reviewWords,aspectTerms,vocab)
  for aspect in range(len(aspectTerms)):
    for word in range(len(reviewWords)):
      sum_row = sum(s_aspect_word[aspect])
      if sum_row > 0:
        reviewMatrix = s_aspect_word[aspect,word] / sum_row
  return reviewWords, reviewMatrix

def createWordMatrix(reviewDataList, vocab, vocabDict, aspectTerms):
  # Ratings analysis and generate review matrix list
  reviewWordsList = []
  for reviews in reviewDataList:
   for review in reviews:
     reviewWords, reviewMatrix = createWMatrixForEachReview(review, vocab, vocabDict, aspectTerms)
     reviewWordsList.append(reviewWords)
     reviewMatrixList.append(reviewMatrix)
  return  reviewWordsList,reviewMatrixList

def generateResults(reviewDataList, reviewWordsList, reviewMatrixList, finalFile):
  f = open(finalFile,"w")
  for reviews in reviewDataList:
    for review in reviews:
      f.write(':'.join([review["ReviewID"], review, str(reviewWordsList[review]), str(reviewMatrixList[review])]) + "\n")

def getVocab():
  #### Step 1: Create vocabulary from json files
  stopWords = genStopwords()
  reviewDataList = getData('HotelData/CleanData') ##Read the json files
  vocab, cnt, vocabDict=createVocab(reviewDataList,stopWords)
  return reviewDataList,vocab, cnt, vocabDict


def runBootstrap():
  #### Step 2. Run model on vocab
  # Loading vocab data from saved file
  reviewDataList, vocab, cnt, vocabDict = getVocab()

  # Get the stemmed term for each aspect from aspect modeling
  aspectTerms = generateAspectTerms(aspectLines, vocabDict)

  #### Run EM algorithm on aspect keywords and save it to a file
  addAspectWords(vocabDict)

  # Create the word matrix for all the reviews
  reviewWordsList,reviewMatrixList = createWordMatrix(reviewDataList, vocab, vocabDict, aspectTerms)
  finalFile = "finalresults.txt"
  # Use the word matrix to generate the results
  generateResults(reviewDataList, reviewWordsList, reviewMatrixList, finalFile)
