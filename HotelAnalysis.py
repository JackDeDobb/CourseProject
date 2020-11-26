# Use Python3
from AnalysisMethods import *
from scipy.special import digamma, gammaln
import json
import nltk
import numpy as np
import os
import ssl


# Dependencies Below
#######################################################################
nltk.download('punkt')
stemmer = nltk.stem.porter.PorterStemmer()
try:
  _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
  pass
else:
  ssl._create_default_https_context = _create_unverified_https_context
#######################################################################


def createVocab(reviewDataList, hotelList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allTerms, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList = [], [], [], [], [], []
  for r in range(len(reviewDataList)):
    for review in reviewDataList[r]['Reviews']:
      parsedWords = parseWords(review['Content'], stopWords)
      reviewFrequency = dict(nltk.FreqDist(parsedWords))
      reviewFreqDictList.append(reviewFrequency)
      reviewList.append(parsedWords)
      reviewIdList.append(review['ReviewID'])
      hotelIdList.append(hotelList[r])
      reviewContentList.append(review['Content'])
      allTerms += parsedWords
  termFrequency = nltk.FreqDist(allTerms)
  vocab, cnt = [], []
  vocabDict = {}
  for k,v in termFrequency.items():
    if v > 5:
      vocab.append(k)
      cnt.append(v)
    else:
      for r in reviewFreqDictList:
        if k in r:
          del r[k]
      for i in range(len(reviewList)):
        reviewList[i] = filter(lambda a: a != k, reviewList[i])
  vocab = np.array(vocab)[np.argsort(vocab)].tolist()
  cnt = np.array(cnt)[np.argsort(vocab)].tolist()
  vocabDict = dict(zip(vocab, range(len(vocab))))
  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList


def generateResults(hotelIdList, reviewIdList, reviewContentList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewList)):
     f.write(':'.join([hotelIdList[i], reviewIdList[i], reviewContentList[i], str(reviewList[i]), str(reviewMatrixList[i])]) + '\n')
  TotalNumOfAnnotatedReviews = 0
  LabelsPerReviewList = []
  for i in range(len(reviewList)):
    for j in range(len(reviewLabelList[i])):
      NumOfAnnotatedReviews=0
      if reviewLabelList[i][j] != -1:
        NumOfAnnotatedReviews += 1 # num of AnnotatedWords in each review
        LabelsPerReviewList.append(NumOfAnnotatedReviews)
      TotalNumOfAnnotatedReviews += NumOfAnnotatedReviews
  print("Total number of hotels =" + str(len(set(hotelIdList))) +"\n")
  print("Total number of reviews =" + str(len(reviewList)) +"\n")
  print("Total number of annotated reviews =" + str(TotalNumOfAnnotatedReviews) +"\n")
  print("Labels per Review=" + str(np.mean(LabelsPerReviewList)) + "+-" + str(np.std(LabelsPerReviewList)) + "\n")


def getData(folder):
  reviewDataList, hotelList = [], []
  for file in os.listdir(folder):
    if file.endswith('.json'):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        reviewDataList.append(json.load(data_file))
        hotelList.append(file.split('.')[0])
  return hotelList, reviewDataList


if __name__ == '__main__':
  stopWords = genStopwords()
  hotelList, reviewDataList = getData('HotelData/testData') # TODO: used TestData for testing ; use CleanData for production # Read the json files
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList = createVocab(reviewDataList, hotelList, stopWords)
  reviewLabelList, reviewMatrixList = runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList)
  generateResults(hotelIdList, reviewIdList, reviewContentList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, 'HotelFinalResults.txt') # Use the word matrix to generate the results
