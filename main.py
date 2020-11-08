##Use Python3

import json
import os
import nltk
import numpy as np
stemmer = nltk.stem.porterPorterStemmer()



def parseWords():
  # Use nltk and stopwords to tokenize words
  pass

def genStopwords():
  # Create stopwords
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))

def readData():
  # Load and read all json files
  pass

def createVocab():
  # Iterate through all the json files data and create vocabulary dictionary having the words and their associated counts
  # Use parseWords to generate the tokenized terms
  # Use nltk.FreqDist to generate term frequqnecies
  pass # return vocabDict

def saveFile(path):
  # Save to file
  pass

def loadFile(path):
  # Load from file
  pass

def generateAspectTerms(aspectLines,vocabDict):
  # Go through each line of the fule and use vocabdict to get the tokenized word
  pass


def addAspectWords(analyzer, p, NumIter,c):
  # Update aspect Words based on Expectation step of EM algorithm
  pass



def getVocab():
  ##### Step 1: Create vocabulary from json files
  genStopwords()  ### generate a list of stopwords
  readData(folder,'json') ## Read the json files
  vocabDict=createVocab() # Create vocabDict

  #### save the vocabulary to a file
  saveFile(path)

def runBootstrap():
  ##### step 2. Run bootstrapping method on vocab
  # Loading vocab data from saved file
  vocabDict=loadFile(path)

  # Load aspect words
  aspectLines=loadFile(aspectWordsFilePath)
  # Get the stemmed term for each aspect from the vocabDict
  aspectTerms=generateAspectTerms(aspectLines,vocabDict)

  #### Run EM algorithm on aspect keywords and save it to a file
  addAspectWords(vocabDict)

  # Create the word matrix for all the reviews
  createWordMatrix()
  # Use the word matrix to generate the results
  generateResults()
