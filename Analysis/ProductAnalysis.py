# Use Python3
from AnalysisMethods import *
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


def createVocab(reviewDataList, productList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allReviewsList, allTerms, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList = [], [], [], [], [], [], [], [], []
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
      productIdList.append(productList[r])
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
  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList


def getData(folder):
  reviewDataList, productList = [], []
  for file in os.listdir(folder):
    if file.endswith('.json'):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        reviewDataList.append(json.load(data_file))
        productList.append(file.split('.')[0])
  return productList, reviewDataList
















def sentenceLabeling(mu, sigma, reviewFreqDictList, vocab, vocabDict): # Update labels
  reviewLabelList = [[] for i in range(len(reviewFreqDictList))]
  for i in range(len(reviewFreqDictList)):
    aspectWeights = np.zeros(shape=(1, len(list(reviewFreqDictList[i].keys()))))
    aspectWeights[0] = np.random.normal(loc=mu, scale=sigma, size=len(list(reviewFreqDictList[i].keys())))
    aspectWeights[0] = aspectWeights[0] / aspectWeights[0].sum(keepdims=1) # Normalize to make row sum=1
    reviewLabels = [-1] * len(list(reviewFreqDictList[i].keys())) # Initialize each review as -1
    reviewLabels[np.where(aspectWeights[0] == max(aspectWeights[0]))[0][0]] = 1 # Change the label to 1 for the word most matching the aspec
    reviewLabelList[i].append(reviewLabels)
  return reviewLabelList


def createWMatrixForEachReview(reviewWordsDict, vocab, vocabDict, reviewLabels): # Generate the matrix for each review
  review = list(reviewWordsDict.keys())
  reviewMatrix = np.zeros(len(review))
  for j in range(len(review)):
    reviewMatrix[j] = reviewWordsDict[review[j]] * reviewLabels[0][j] # Get the review rating
  reviewMatrix = (reviewMatrix - reviewMatrix.min(0)) / reviewMatrix.ptp(0)
  #TODO: for some reason, we are getting the same values of rows in each column.
  #      Here, we are multiplying the label of each word with it's count in the revie and creating a matrix.
  #      Thus, theoretically, we should not be getting that. Please help debug. Some thing is off in implementation in
  #      sentence Labelling or the way this matrix is created.
  return reviewMatrix


def createWordMatrix(reviewFreqDictList, vocab, vocabDict, reviewLabelList): # Ratings analysis and generate review matrix list
  reviewMatrixList = []
  for i in range(len(reviewFreqDictList)):
    reviewMatrix = createWMatrixForEachReview(reviewFreqDictList[i], vocab, vocabDict, reviewLabelList[i])
    reviewMatrixList.append(reviewMatrix) # TODO: should this flattened?
  return reviewMatrixList

def getOverallRatingsForWords(reviewFreqDictList, reviewMatrixList):
  positiveWordList, negativeWordList = [], []
  for i in range(len(reviewMatrixList)):
    BestSentimentIndex=reviewMatrixList[i].argmax(axis=0)
    WorstSentimentIndex=reviewMatrixList[i].argmin(axis=0)
    positiveWordList.append(list(reviewFreqDictList[i].keys())[BestSentimentIndex])
    negativeWordList.append(list(reviewFreqDictList[i].keys())[WorstSentimentIndex])
  return positiveWordList, negativeWordList

def generatePredictedAspects(reviewFreqDictList,reviewMatrixList):
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





def runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList, allReviewsList):
  mu, sigma = generateAspectParameters(reviewFreqDictList, vocabDict) # Aspect modeling to get parameters
  reviewLabelList = sentenceLabeling(mu, sigma, reviewFreqDictList, vocab, vocabDict) # Create aspects and get labels from aspect terms on reviews
  reviewMatrixList = createWordMatrix(reviewFreqDictList, vocab, vocabDict, reviewLabelList) # Create the word matrix for all the reviews
  positiveWordList, negativeWordList = getOverallRatingsForWords(reviewFreqDictList, reviewMatrixList)
  predList = generatePredictedAspects(reviewFreqDictList, reviewMatrixList)
  totalMse, totalPearson = getStats(predList, allReviewsList)
  return reviewLabelList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson
























if __name__ == '__main__':
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  cleanDataLocation = '/'.join([currDirectoryOfScript, '..', 'Data', 'ProductData', 'testData']) # TODO: switch from testData to CleanData
  resultsLocation = '/'.join([currDirectoryOfScript, '..', 'Results', 'ProductFinalResults.txt'])
  stopWords = genStopwords()
  productList, reviewDataList = getData(cleanDataLocation)
  print('DEBUG: getData')
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList = createVocab(reviewDataList, productList, stopWords)
  print('DEBUG: createVocab')
  reviewLabelList, reviewMatrixList,positiveWordList, negativeWordList, totalMse,totalPearson = runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList, allReviewsList)
  print('DEBUG: run algo')
  generateResults(productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson, resultsLocation) # Use the word matrix to generate the results
