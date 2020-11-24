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


def createVocab(reviewDataList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allTerms, reviewList = [], []
  for reviewData in reviewDataList:
    for review in reviewData['Reviews']:
      parseWordsInReview = []
      parsedWords = parseWords(review['Content'], stopWords)
      reviewList.append(parsedWords)
      for parseWord in parsedWords:
        parseWordsInReview = parseWord + parseWordsInReview
      allTerms += parseWordsInReview
  termFrequency = nltk.FreqDist(allTerms)
  vocab, cnt = [], []
  vocabDict = {}
  for k,v in termFrequency.items():
    if v > 5:
      vocab.append(k)
      cnt.append(v)
  vocab = np.array(vocab)[np.argsort(vocab)].tolist()
  cnt = np.array(cnt)[np.argsort(vocab)].tolist()
  vocabDict = dict(zip(vocab, range(len(vocab))))
  return vocab, cnt, vocabDict, reviewList


def generateResults(hotelList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewDataList)):
    hotelId = hotelList[i]
    for review in reviewDataList[i]:
      f.write(':'.join([hotelId, review['ReviewID'], review, str(reviewWordsList[review]), str(reviewMatrixList[review])]) + '\n')
  TotalNumOfReviews, TotalNumOfAnnotatedReviews = 0, 0
  LabelsPerReviewList = []
  for i in range(len(reviewDataList)):
    TotalNumOfReviews += len(reviewDataList[i])
    for j in range(len(reviewDataList[i])):
      LabelsPerReviewList[j] = 0
      for label in reviewLabelList[j]:
        if label!=-1:
          LabelsPerReviewList[j] += 1 # num of AnnotatedWords in each review
      TotalNumOfAnnotatedReviews += 1
  print("Total number of hotels =" + str(len(hotelList)) +"\n")
  print("Total number of reviews =" + str(TotalNumOfReviews) +"\n")
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


def getVocab():
  stopWords = genStopwords()
  hotelList, reviewDataList = getData('HotelData/CleanData') # Read the json files
  vocab, cnt, vocabDict, reviewList = createVocab(reviewDataList, stopWords)
  return hotelList, reviewDataList, vocab, cnt, vocabDict, reviewList


if __name__ == '__main__':
  hotelList, reviewDataList, vocab, cnt, vocabDict, reviewList = getVocab() # Loading vocab data from saved file
  reviewLabelList, reviewWordsList, reviewMatrixList = runAlgorithm(vocab, cnt, vocabDict, reviewList)
  generateResults(hotelList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, 'HotelFinalResults.txt') # Use the word matrix to generate the results
