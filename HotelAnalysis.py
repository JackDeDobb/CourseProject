# Use Python3
from AnalysisMethods import *
from scipy.special import digamma, gammaln
import json
import nltk
import numpy as np
import os
import random
import ssl
import string


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
  allTerms = []
  reviewList = []
  for reviewData in reviewDataList:
    for review in reviewData['Reviews']:
      parseWordsInReview = []
      parsedWords = parseWords(review['Content'], stopWords)
      reviewList.append(parsedWords)
      for parseWord in parsedWords:
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
  return vocab, cnt, vocabDict, reviewList

def generateAspectParameters(reviewList, vocabDict):
  #Aspect modeling
  k = 4  # nbr of latent states z
  M = len(reviewList)  # nbr of reviews
  initMu = 0.0
  initSigma = 0.0
  initPhi, initEta, initGamma, initEpsilon, initLambda, initSigmaSq = initializeParameters(reviewList, vocabDict, M, k)
  for d in range(0,M):
    initMu += initLambda[d]
  initMu = initMu / M
  for d in range(0, M):
    initSigma += (initLambda[d] - initMu)**2 + initSigmaSq[d]**2
  initSigma = initSigma / M
  phi, eta, gamma, epsilon, mu, sigma, likelihood = EM(initPhi, initEta, initGamma, initEpsilon, initLambda, initSigmaSq, initMu, initSigma, reviewList, vocabDict, M,k)
  return mu, sigma

def sentenceLabeling(mu, sigma, reviewList, vocab, vocabDict): # Update labels
  reviewWordsList = []
  reviewLabelList = []
  for i in range(len(reviewList)):
    review = reviewList[i]
    aspectWeights = aspectTerms[i]
    reviewWords = parseWordsForSentence(review, vocab, vocabDict)
    reviewLabels = [-1] * len(reviewWords) # Initialize each review as -1
    reviewWordsList.append(reviewWords)
    aspectWeights = np.random.normal(loc=mu, scale=sigma, size=len(reviewWords))
    aspectWeights = aspectWeights / aspectWeights.sum(axis=0, keepdims=1) # Normalize to make row sum=1
    count = max(aspectWeights)
    reviewLabels[aspectWeights.index[count]] = 1 # Change the label to 1 for the word most matching the aspect
    reviewLabelList.append(reviewLabels)
  return reviewWordsList, reviewLabelList

def createWMatrixForEachReview(reviewWords, review, vocab, vocabDict, reviewLabels):
  # Generate the matrix for each review
  reviewMatrix = np.zeros((len(reviewLabels),len(reviewWords)))
  for i in range(len(reviewLabels)):
    for j in range(len(reviewWords)):
      reviewMatrix[i,j]=reviewWords[i]*reviewLabels[j] #Get the review rating
    reviewMatrix[i] =reviewMatrix[i]/reviewMatrix[i].sum(axis=0,keepdims=1)   #Normalize to make row sum=1
  return reviewMatrix

def createWordMatrix(reviewWordsList, reviewList, vocab, vocabDict, reviewLabelList):
  # Ratings analysis and generate review matrix list
  reviewMatrixList = []
  for i in range(len(reviewList)):
    reviewMatrix = createWMatrixForEachReview(reviewWordsList[i], reviews[i], vocab, vocabDict, reviewLabelList[i])
    reviewMatrixList.append(reviewMatrix)
  return reviewMatrixList

def generateResults(hotelList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewDataList)):
    hotelId = hotelList[i]
    for review in reviewDataList[i]:
      f.write(':'.join([hotelId, review["ReviewID"], review, str(reviewWordsList[review]), str(reviewMatrixList[review])]) + "\n")
  TotalNumOfHotels = len(hotelList)
  TotalNumOfReviews = 0
  TotalNumOfAnnotatedReviews = 0
  LabelsPerReviewList = []
  for i in range(len(reviewDataList)):
    TotalNumOfReviews = TotalNumOfReviews + len(reviewDataList[i])
    for j in range(len(reviewDataList[i])):
      LabelsPerReviewList[j]=0
      for label in reviewLabelList[j]:
        if label!=-1:
          LabelsPerReviewList[j] += 1 # num of AnnotatedWords in each review
      TotalNumOfAnnotatedReviews = TotalNumOfAnnotatedReviews + 1
  print("Total number of hotels =" + str(TotalNumOfHotels) +"\n")
  print("Total number of reviews =" + str(TotalNumOfReviews) +"\n")
  print("Total number of annotated reviews =" + str(TotalNumOfAnnotatedReviews) +"\n")
  print("Labels per Review=" + str(np.mean(LabelsPerReviewList)) + "+-" + str(np.std(LabelsPerReviewList)) + "\n")

def getData(folder):
  reviewDataList = []
  hotelList = []
  for file in os.listdir(folder):
    if file.endswith(".json"):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        data = json.load(data_file)
        hotelList.append(file.split('.')[0])
        reviewDataList.append(data)
  return hotelList, reviewDataList

def getVocab():
  stopWords = genStopwords()
  hotelList, reviewDataList = getData('HotelData/CleanData') # Read the json files
  vocab, cnt, vocabDict, reviewList = createVocab(reviewDataList, stopWords)
  return hotelList, reviewDataList, vocab, cnt, vocabDict, reviewList


if __name__ == '__main__':
  hotelList, reviewDataList, vocab, cnt, vocabDict, reviewList = getVocab() # Loading vocab data from saved file
  mu, sigma = generateAspectParameters(reviewList, vocabDict) # aspect modeling to get parameters
  reviewWordsList, reviewLabelList = sentenceLabeling(mu, sigma, reviewList, vocab, vocabDict) # Create aspects and get labels from aspect terms on reviews
  reviewMatrixList = createWordMatrix(reviewWordsList,reviewList, vocab, vocabDict, reviewLabelList) # Create the word matrix for all the reviews  
  generateResults(hotelList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, 'finalresults.txt') # Use the word matrix to generate the results
