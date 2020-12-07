# Use Python3
from AnalysisMethods import *
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


def createVocab(reviewDataList, itemList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allReviewsList, allTerms, reviewList, reviewFreqDictList, itemIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList = [], [], [], [], [], [], [], [], []
  print(range(len(reviewDataList)))
  for r in range(len(reviewDataList)):
    if (r % 300 == 0):
      print('r = ' + str(r))
    for review in reviewDataList[r]:
      parsedWords = parseWords(review['fullText'], stopWords)
      reviewFrequency = dict(nltk.FreqDist(parsedWords))
      reviewFreqDictList.append(reviewFrequency)
      reviewList.append(parsedWords)
      reviewIdList.append(review['reviewId'])
      allReviewsList.append(review['rating'])
      itemIdList.append(itemList[r])
      reviewContentList.append(review['fullText'])
      reviewRatingList.append(review['rating'])
      reviewAuthorList.append(review['author'])
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
  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, itemIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList


def createWMatrixForEachReview(reviewWordsDict, reviewLabels): # Generate the matrix for each review
  review = list(reviewWordsDict.keys())
  reviewMatrix = np.zeros(len(review))
  for j in range(len(review)):
    reviewMatrix[j] = reviewWordsDict[review[j]] * reviewLabels[0][j] # Get the review rating
  reviewMatrix = (reviewMatrix - reviewMatrix.min(0)) / reviewMatrix.ptp(0)
  return reviewMatrix


def createWordMatrix(reviewFreqDictList, reviewLabelList): # Ratings analysis and generate review matrix list
  reviewMatrixList = []
  for i in range(len(reviewFreqDictList)):
    reviewMatrixList.append(createWMatrixForEachReview(reviewFreqDictList[i], reviewLabelList[i]))
  return reviewMatrixList


def generatePredictedAspects(reviewFreqDictList, reviewMatrixList):
  predList = []
  for j in range(len(reviewMatrixList)):
    predReviews = 0
    for k in range(len(reviewMatrixList[j])):
      review = list(reviewFreqDictList[j].keys())
      predReviews += reviewFreqDictList[j][review[k]]*reviewMatrixList[j][k]
    predReviews = predReviews/len(reviewMatrixList[j])
    predList.append(predReviews)
  predList = [float(i) * 5 / max(predList) for i in predList]
  return predList


def runAlgorithm(vocabDict, reviewFreqDictList, allReviewsList):
  mu, sigma = generateAspectParameters(reviewFreqDictList, vocabDict) # Aspect modeling to get parameters
  reviewLabelList = sentenceLabeling(mu, sigma, reviewFreqDictList, 1) # Create aspects and get labels from aspect terms on reviews
  reviewMatrixList = createWordMatrix(reviewFreqDictList, reviewLabelList) # Create the word matrix for all the reviews
  positiveWordList, negativeWordList = getOverallRatingsForWords(reviewFreqDictList, reviewMatrixList)
  predList = generatePredictedAspects(reviewFreqDictList, reviewMatrixList)
  totalMse, totalPearson = getStats(predList, allReviewsList)
  return reviewLabelList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson


if __name__ == '__main__':
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  cleanDataLocation = '/'.join([currDirectoryOfScript, '..', 'Data', 'ProductData', 'testData']) # TODO: switch from testData to CleanData
  resultsLocation = '/'.join([currDirectoryOfScript, '..', 'Results', 'ProductFinalResults.txt'])
  stopWords = genStopwords()
  itemList, reviewDataList = getData(cleanDataLocation)
  print('DEBUG: getData')
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, itemIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList = createVocab(reviewDataList, itemList, stopWords)
  print('DEBUG: createVocab')
  reviewLabelList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson = runAlgorithm(vocabDict, reviewFreqDictList, allReviewsList)
  print('DEBUG: run algo')
  generateResults(itemIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson, resultsLocation) # Use the word matrix to generate the results
