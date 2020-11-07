import json
import os
import string


currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
hotelDataFilesDirectory = '/'.join([currDirectoryOfScript, 'HotelData', 'RawData'])
hotelDataFileNames = os.listdir(hotelDataFilesDirectory)

stopWords = ''
with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
  stopWords = set(json.load(stopWords))

hotelIDToDataMapping = {}
for hotelDataFileName in hotelDataFileNames:
  with open('/'.join([hotelDataFilesDirectory, hotelDataFileName])) as hotelDataFile:
    hotelData = json.load(hotelDataFile)
    hotelID = hotelData['HotelInfo']['HotelID']
    hotelIDToDataMapping[hotelID] = hotelData


reviewCounter = 0
# Part 1:   Remove reviews with any missing aspect rating or document
#           length less than 50 words
# Part 2:   Convert all the words into lower cases
# Part 3.1: Remove punctuation and stop words
ratingCategories = ['Service', 'Cleanliness', 'Overall', 'Value', 'Sleep Quality', 'Rooms', 'Location']
for (hotelID, hotelData) in list(hotelIDToDataMapping.items()):
  hotelReviews = hotelData['Reviews']
  filteredHotelReviews = []
  for hotelReview in hotelReviews:
    if 'Content' not in hotelReview or len(hotelReview['Content'].split()) < 50:
      continue

    try:
      [float(hotelReview['Ratings'][ratingCategory]) for ratingCategory in ratingCategories]
    except:
      continue

    hotelReview['Content'] = hotelReview['Content'].lower()
    hotelReview['Content'] = hotelReview['Content'].translate(str.maketrans('', '', string.punctuation))
    hotelReview['Content'] = ' '.join(list(filter(lambda x: x not in stopWords, hotelReview['Content'].split())))
    filteredHotelReviews.append(hotelReview)

  hotelIDToDataMapping[hotelID]['Reviews'] = filteredHotelReviews
  if len(filteredHotelReviews) == 0:
    del hotelIDToDataMapping[hotelID]
  reviewCounter += len(filteredHotelReviews)

print('hotelCounter: ' + str(len(hotelIDToDataMapping)))
print('reviewCounter: ' + str(reviewCounter))
#print(hotelIDToDataMapping)
