import json
import os


currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
stopWords = ''
with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
  stopWords = set(json.load(stopWords))


print(stopWords)
