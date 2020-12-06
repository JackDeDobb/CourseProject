# Use Python3
from ProductAnalysisMethods import *
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


def createVocab(reviewDataList, productList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allTerms, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList = [], [], [], [], [], [], [], []
  allReviewsList=[]
  for r in range(len(reviewDataList)):
    for review in reviewDataList[r]:#['Reviews']:
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
  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList,reviewRatingList, reviewAuthorList, allReviewsList


def generateResults(productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewList)):
     f.write(':'.join([productIdList[i], reviewIdList[i], reviewContentList[i], str(reviewList[i]), str(reviewMatrixList[i])]) + '\n')
  TotalNumOfAnnotatedReviews,TotalLengthOfReviews = 0, 0
  LabelsPerReviewList = []
  for i in range(len(reviewList)):
    TotalLengthOfReviews+=len(reviewContentList[i])
    for j in range(len(reviewLabelList[i])):
      NumOfAnnotatedReviews=0
      if reviewLabelList[i][j] != -1:
        NumOfAnnotatedReviews += 1 # num of AnnotatedWords in each review
        LabelsPerReviewList.append(NumOfAnnotatedReviews)
      TotalNumOfAnnotatedReviews += NumOfAnnotatedReviews
  print("Total number of items =" + str(len(set(productIdList))) +"\n")
  print("Total number of reviews =" + str(len(reviewList)) +"\n")
  print("Total number of annotated reviews =" + str(TotalNumOfAnnotatedReviews) +"\n")
  print("Labels per Review=" + str(np.mean(LabelsPerReviewList)) + "+-" + str(np.std(LabelsPerReviewList)) + "\n")
  print("Total number of reviewers ="+ str(len(set(reviewAuthorList))) +"\n")
  print("Average length of review ="+ str(TotalLengthOfReviews/len(reviewList)) +"\n")
  print("Ratings of review ="+ str(np.mean(reviewRatingList))+"+-"+str(np.std(reviewRatingList)) +"\n")
  print("High Overall Ratings =" +str(sorted(dict(nltk.FreqDist(positiveWordList)).items(), key=lambda item: item[1], reverse=True)[:30]))
  print("Low Overall Ratings =" +str(sorted(dict(nltk.FreqDist(negativeWordList)).items(), key=lambda item: item[1], reverse=True)[:30]))
  print("Total MSE ="+str(totalMse))
  print("Total Pearson ="+ str(totalPearson))


def getData(folder):
  reviewDataList, productList = [], []
  for file in os.listdir(folder):
    if file.endswith('.json'):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        reviewDataList.append(json.load(data_file))
        productList.append(file.split('.')[0])
  return productList, reviewDataList


if __name__ == '__main__':
  stopWords = genStopwords()
  productList, reviewDataList = getData('ProductData/testData') # Read the json files
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList = createVocab(reviewDataList, productList, stopWords)
  reviewLabelList, reviewMatrixList,positiveWordList, negativeWordList, totalMse,totalPearson = runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList, allReviewsList)
  generateResults(productIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, 'ProductFinalResults.txt') # Use the word matrix to generate the results
