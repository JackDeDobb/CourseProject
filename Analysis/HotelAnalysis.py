# Use Python3
from HotelAnalysisMethods import *
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
  allTerms, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList,reviewRatingList, reviewAuthorList  = [], [], [], [], [], [], [], []
  #serviceList, cleanlinessList,valueList, sleepQualityList,roomsList,locationList= [], [], [], [], [], []
  allReviewsList=[]
  print(range(len(reviewDataList)))
  for r in range(len(reviewDataList)):
    if (r % 300 == 0):
      print('r = ' + str(r))
    for review in reviewDataList[r]['Reviews']:
      parsedWords = parseWords(review['Content'], stopWords)
      reviewFrequency = dict(nltk.FreqDist(parsedWords))
      reviewFreqDictList.append(reviewFrequency)
      reviewList.append(parsedWords)
      reviewIdList.append(review['ReviewID'])
      hotelIdList.append(hotelList[r])
      reviewContentList.append(review['Content'])
      allReviewsList.append(review['Ratings']['Service'])
      allReviewsList.append(review['Ratings']['Cleanliness'])
      reviewRatingList.append(review['Ratings']['Overall'])
      allReviewsList.append(review['Ratings']['Overall'])
      allReviewsList.append(review['Ratings']['Value'])
      allReviewsList.append(review['Ratings']['Sleep Quality'])
      allReviewsList.append(review['Ratings']['Rooms'])
      allReviewsList.append(review['Ratings']['Location'])
      reviewAuthorList.append(review['Author'])
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

  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList,reviewRatingList, reviewAuthorList, allReviewsList


def generateResults(hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewList)):
     f.write(':'.join([hotelIdList[i], reviewIdList[i], reviewContentList[i], str(reviewList[i]), str(reviewMatrixList[i])]) + '\n')
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
  print("Total number of items =" + str(len(set(hotelIdList))) +"\n")
  print("Total number of reviews =" + str(len(reviewList)) +"\n")
  print("Total number of annotated reviews =" + str(TotalNumOfAnnotatedReviews) +"\n")
  print("Labels per Review =" + str(np.mean(LabelsPerReviewList)) + "+-" + str(np.std(LabelsPerReviewList)) + "\n")
  print("Total number of reviewers ="+ str(len(set(reviewAuthorList))) +"\n")
  print("Average length of review ="+ str(TotalLengthOfReviews/len(reviewList)) +"\n")
  print("Ratings of review ="+ str(np.mean(reviewRatingList))+"+-"+str(np.std(reviewRatingList)) +"\n")
  print("High Overall Ratings =" +str(sorted(dict(nltk.FreqDist(positiveWordList)).items(), key=lambda item: item[1], reverse=True)[:30]))
  print("Low Overall Ratings =" +str(sorted(dict(nltk.FreqDist(negativeWordList)).items(), key=lambda item: item[1], reverse=True)[:30]))
  print("Total MSE ="+str(totalMse))
  print("Total Pearson ="+ str(totalPearson))

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
  print('DEBUG: stop words')
  hotelList, reviewDataList = getData('Data/HotelData/testData') # TODO: used TestData for testing ; use CleanData for production # Read the json files
  print('DEBUG: getData')
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList = createVocab(reviewDataList, hotelList, stopWords)
  print('DEBUG: createVocab')
  reviewLabelList, reviewMatrixList,positiveWordList, negativeWordList, totalMse,totalPearson = runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList, allReviewsList)
  print('DEBUG: run algo')
  generateResults(hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, 'Results/HotelFinalResults.txt') # Use the word matrix to generate the results
