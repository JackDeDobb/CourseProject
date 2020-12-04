from collections import defaultdict
from datetime import datetime
import json
import os
import shutil
import string


currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
productDataCleanFilesDirectory = '/'.join([currDirectoryOfScript, 'ProductData', 'CleanData'])
productDataRawFile = '/'.join([currDirectoryOfScript, 'ProductData', 'RawData', 'amazon_mp3.txt'])

stopWords = ''
with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
  stopWords = set(json.load(stopWords))

dataAttributes = {
  'productName': 3,
  'title': 4,
  'author': 5,
  'createDate': 6,
  'fullText': 8,
  'rating': 9,
  'helpfulNum': 12,
  'totalNum': 13,
  'commentNum': 14,
  'webUrl': 16,
  'htmlPath': 17
}
# Parse Text Data File into dictionary of json files
productIDToDataMapping = defaultdict(list)
allReviews = ''
with open(productDataRawFile) as productDataRawFile:
  allReviews = productDataRawFile.read().split('#####')
  for review in allReviews[1:]:
    reviewAttributes = review.split('\n')[1:]
    jsonObj = {}
    for (dataAttribute, reviewAttributesIndex) in dataAttributes.items():
      try:
        jsonObj[dataAttribute] = ':'.join(reviewAttributes[reviewAttributesIndex].split(':')[1:])
      except:
        pass
    productID = jsonObj['webUrl'].split('/')[5]
    productIDToDataMapping[productID].append(jsonObj)


# Part 1:   Remove reviews with any missing aspect rating or document
#           length less than 50 words
# Part 2:   Convert all the words into lower cases
# Part 3.1: Remove punctuation and stop words
# Part 3.2: Gather -- Remove words occuring in less than 10 reviews in the collection
ratingCategories = ['rating']
otherCategories = ['helpfulNum', 'totalNum', 'commentNum']
wordToReviewOccurencesMapping = defaultdict(int)
for (productID, productReviews) in productIDToDataMapping.items():
  print('first pass: ' + productID)
  filteredProductReviews = []
  for productReview in productReviews:
    if len(productReview['fullText'].split()) < 50:
      continue
    try:
      for ratingCategory in ratingCategories:
        productReview[ratingCategory] = float(productReview[ratingCategory])
      for otherCategory in otherCategories:
        productReview[otherCategory] = int(productReview[otherCategory])
    except:
      continue

    if productReview['createDate'] != '':
      productReview['createDate'] = datetime.strptime(productReview['createDate'], '%a %b %d %H:%M:%S CST %Y')
    else:
      del productReview['createDate']
    productReview['fullText'] = productReview['fullText'].lower()
    productReview['fullText'] = productReview['fullText'].translate(str.maketrans('', '', string.punctuation))
    productReview['fullText'] = list(filter(lambda x: x not in stopWords, productReview['fullText'].split()))
    filteredProductReviews.append(productReview)

    for word in set(productReview['fullText']):
      wordToReviewOccurencesMapping[word] += 1

    productIDToDataMapping[productID] = filteredProductReviews
    if len(filteredProductReviews) == 0:
      del productIDToDataMapping[productID]


# Part 3.2: Apply -- Remove words occuring in less than 10 reviews in the collection ()
wordToReviewOccurencesMapping = { k: v for k, v in wordToReviewOccurencesMapping.items() if v >= 10 }
wordSetToKeep = set(wordToReviewOccurencesMapping.keys())
for (productID, productReviews) in list(productIDToDataMapping.items()):
  print('second pass: ' + productID)
  filteredProductReviews = []
  for productReview in productReviews:
    productReview['fullText'] = ' '.join(list(filter(lambda x: x in wordSetToKeep, productReview['fullText'])))
    filteredProductReviews.append(productReview)

  productIDToDataMapping[productID]= filteredProductReviews
  if len(filteredProductReviews) == 0:
    del productIDToDataMapping[productID]


if os.path.isdir(productDataCleanFilesDirectory):
  shutil.rmtree(productDataCleanFilesDirectory)
os.mkdir(productDataCleanFilesDirectory)
for (productID, productReviews) in list(productIDToDataMapping.items()):
  with open('/'.join([productDataCleanFilesDirectory, productID + '.json']), 'w+') as cleanDataFile:
    print('exporting file: ' + productID + '.json')
    json.dump(productReviews, cleanDataFile, default=str, indent=2)
