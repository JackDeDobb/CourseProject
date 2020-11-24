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
  # Fill in method here
  pass


def generateResults(productList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, finalFile):
  # Fill in method here
  pass


def getData(folder):
  # Fill in method here
  pass


if __name__ == '__main__':
  stopWords = genStopwords()
  productList, reviewDataList = getData('ProductData/CleanData') # Read the json files
  vocab, cnt, vocabDict, reviewList = createVocab(reviewDataList, stopWords)
  reviewLabelList, reviewWordsList, reviewMatrixList = runAlgorithm(vocab, cnt, vocabDict, reviewList)
  generateResults(productList, reviewDataList, reviewLabelList, reviewWordsList, reviewMatrixList, 'ProductFinalResults.txt') # Use the word matrix to generate the results
