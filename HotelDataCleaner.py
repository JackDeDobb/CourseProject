from collections import defaultdict
import json
import os
import shutil
import string


currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
hotelDataRawFilesDirectory = '/'.join([currDirectoryOfScript, 'HotelData', 'RawData'])
hotelDataCleanFilesDirectory = '/'.join([currDirectoryOfScript, 'HotelData', 'CleanData'])
hotelDataFileNames = os.listdir(hotelDataRawFilesDirectory)

stopWords = ''
with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
  stopWords = set(json.load(stopWords))

hotelIDToDataMapping = {}
for hotelDataFileName in hotelDataFileNames[0:10]:
  with open('/'.join([hotelDataRawFilesDirectory, hotelDataFileName])) as hotelDataFile:
    print('loading in file: ' + hotelDataFileName)
    hotelData = json.load(hotelDataFile)
    hotelID = hotelData['HotelInfo']['HotelID']
    hotelIDToDataMapping[hotelID] = hotelData


# Part 1:   Remove reviews with any missing aspect rating or document
#           length less than 50 words
# Part 2:   Convert all the words into lower cases
# Part 3.1: Remove punctuation and stop words
# Part 3.2: Gather -- Remove words occuring in less than 10 reviews in the collection
ratingCategories = ['Service', 'Cleanliness', 'Overall', 'Value', 'Sleep Quality', 'Rooms', 'Location']
wordToReviewOccurencesMapping = defaultdict(int)
for (hotelID, hotelData) in list(hotelIDToDataMapping.items()):
  print('first pass: ' + hotelID)
  filteredHotelReviews = []
  for hotelReview in hotelData['Reviews']:
    if 'Content' not in hotelReview or len(hotelReview['Content'].split()) < 50:
      continue
    try:
      [float(hotelReview['Ratings'][ratingCategory]) for ratingCategory in ratingCategories]
    except:
      continue

    hotelReview['Content'] = hotelReview['Content'].lower()
    hotelReview['Content'] = hotelReview['Content'].translate(str.maketrans('', '', string.punctuation))
    hotelReview['Content'] = list(filter(lambda x: x not in stopWords, hotelReview['Content'].split()))
    filteredHotelReviews.append(hotelReview)

    for word in set(hotelReview['Content']):
      wordToReviewOccurencesMapping[word] += 1

  hotelIDToDataMapping[hotelID]['Reviews'] = filteredHotelReviews
  if len(filteredHotelReviews) == 0:
    del hotelIDToDataMapping[hotelID]


# Part 3.2: Apply -- Remove words occuring in less than 10 reviews in the collection ()
wordToReviewOccurencesMapping = { k: v for k, v in wordToReviewOccurencesMapping.items() if v >= 10 }
wordSetToKeep = set(wordToReviewOccurencesMapping.keys())
for (hotelID, hotelData) in list(hotelIDToDataMapping.items()):
  print('second pass: ' + hotelID)
  filteredHotelReviews = []
  for hotelReview in hotelData['Reviews']:
    hotelReview['Content'] = ' '.join(list(filter(lambda x: x in wordSetToKeep, hotelReview['Content'])))
    filteredHotelReviews.append(hotelReview)

  hotelIDToDataMapping[hotelID]['Reviews'] = filteredHotelReviews
  if len(filteredHotelReviews) == 0:
    del hotelIDToDataMapping[hotelID]


if os.path.isdir(hotelDataCleanFilesDirectory):
  shutil.rmtree(hotelDataCleanFilesDirectory)
os.mkdir(hotelDataCleanFilesDirectory)
for (hotelID, hotelData) in list(hotelIDToDataMapping.items()):
  with open('/'.join([hotelDataCleanFilesDirectory, hotelID + '.json']), 'w+') as cleanDataFile:
    json.dump(hotelData, cleanDataFile)