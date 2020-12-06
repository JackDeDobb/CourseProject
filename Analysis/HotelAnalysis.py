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


def createVocab(reviewDataList, hotelList, stopWords):
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  allReviewsList, allTerms, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList  = [], [], [], [], [], [], [], [], []
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

  return vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList

def getData(folder):
  reviewDataList, hotelList = [], []
  for file in os.listdir(folder):
    if file.endswith('.json'):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        reviewDataList.append(json.load(data_file))
        hotelList.append(file.split('.')[0])
  return hotelList, reviewDataList


if __name__ == '__main__':
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  cleanDataLocation = '/'.join([currDirectoryOfScript, '..', 'Data', 'HotelData', 'testData']) # TODO: switch from testData to CleanData
  resultsLocation = '/'.join([currDirectoryOfScript, '..', 'Results', 'HotelFinalResults.txt'])
  stopWords = genStopwords()
  hotelList, reviewDataList = getData(cleanDataLocation)
  print('DEBUG: getData')
  vocab, cnt, vocabDict, reviewList, reviewFreqDictList, hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, allReviewsList = createVocab(reviewDataList, hotelList, stopWords)
  print('DEBUG: createVocab')
  reviewLabelList, reviewMatrixList,positiveWordList, negativeWordList, totalMse, totalPearson = runAlgorithm(vocab, cnt, vocabDict, reviewList, reviewFreqDictList, allReviewsList)
  print('DEBUG: run algo')
  generateResults(hotelIdList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson, resultsLocation) # Use the word matrix to generate the results
