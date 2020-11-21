# Use Python3
from scipy.special import digamma, gammaln
import json
import nltk
import numpy as np
import os
import random
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



def parseWordsForSentence(content, vocab, vocabDict):
  # Use nltk and stopwords to tokenize words for  each review
  tokenizedWords = []
  sentences = nltk.sent_tokenize(content)
  for sent in sentences:
    words = nltk.word_tokenize(sent)
    stemmedWordsInVocab = [stemmer.stem(w.lower()) for w in words if stemmer.stem(w.lower()) in vocabDict]
    words = [vocabDict.get(w) for w in stemmedWordsInVocab]
    if len(words) > 0:
      tokenizedWords.append(words)
  return tokenizedWords

def parseWords(content, stopWords):
  # Use nltk and stopwords to tokenize words
  tokenizedWords = []
  sentences = nltk.sent_tokenize(content)
  for sent in sentences:
    words = nltk.word_tokenize(sent)
    stemmedWords = [stemmer.stem(w.lower()) for w in words if w not in string.punctuation]
    stemmedWordsWithoutStopwords = [v for v in stemmedWords if v not in stopWords] # Remove stopwords
    if len(stemmedWordsWithoutStopwords) > 0:
      tokenizedWords.append(stemmedWordsWithoutStopwords)
  return tokenizedWords

def genStopwords():
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))

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

def initializeParameters(reviewList, vocabDict, M, k):
  phi = []
  eta = np.zeros([M, k])
  lmbda = []
  sigmaSq = []
  gamma = np.ones([M, k])
  for m in range(0, M):
    wordsInDoc = reviewList[m]
    N = len(wordsInDoc)
    phi = np.ones([N, k]) * 1 / float(k)
    for i in range(0, k):
      eta[m, i] = gamma[m, i] + N / float(k)
      phi.append(phi)
      lmbda.append(phi)
      sigmaSq.append(np.random.rand())
      m += 1
  epsilon = np.zeros([k, len(vocabDict)])
  for i in range(0, k):
    tmp = np.random.uniform(0, 1, len(vocabDict))
    epsilon[i,:] = tmp / np.sum(tmp)
  return phi, eta, gamma, epsilon, lmbda,sigmaSq

def calcLikelihood(phi, eta, gamma, epsilon, review, vocabDict, k):
  likelihood = 0.0
  V = len(vocabDict)
  N = len(review)

  gammaSum = 0.0
  phiEtaSum = 0.0
  phiLogEpsilonSum = 0.0
  entropySum = 0.0
  etaSum = 0.0

  gammaSum += gammaln(np.sum(alpha))
  etaSum -= gammaln(np.sum(gamma))
  for i in range(0,k):
    gammaSum += -gammaln(gamma[i]) + (gamma[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta)))
    for n in range(0,N):
      if Phi[n,i] > 0:
        indicator = np.sum(np.in1d(len(vocabDict), review[n]))
        phiEtaSum += phi[n,i] * (digamma(eta[i]) - digamma(np.sum(eta[:])))
        entropySum += phi[n,i] * np.log(phi[n,i])
        for j in range(0,V):
          if Beta[i,j] > 0:
            phiLogEpsilonSum += phi[n,i] * indicator * np.log(epsilon[i,j])
    etaSum += gammaln(eta[i]) - (eta[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta[:])))

  likelihood += (gammaSum + phiEtaSum + phiLogEpsilonSum - etaSum - entropySum)
  return likelihood

def EStep(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewList, vocabDict, k, M):
  print('E-step')
  likelihood = 0.0
  newMu = 0.0
  newSigma = 0.0
  newLmbda = []
  newSigmaSq = []
  convergence = np.zeros(M)
  for d in range(0,M):
    words = reviewList[d]
    N = len(words)
    p = phi[d]
    counter = 0
    while convergence[d] == 0 and d < len(convergence):
      oldPhi = p
      p = np.zeros([N,k])
      oldEta = eta[d, :]
      for n in range(0,N):
        word = words[n]
        vocabIdx = list(vocabDict).index(word)
        if len(vocabIdx[0]) > 0:
          for i in range(0,k):
            e = epsilon[i, vocabIdx]
            p[n, i] = e[0][0] * np.exp(digamma(eta[d,i]) - digamma(np.sum(eta[d,:])))
          p[n,:] = p[n,:] / np.sum(p[n,:])
      eta[d,:] = gamma[d,:] + np.sum(p, axis=0)
      newLmbda[d] = 0.5 * (lmbda[d] - mu)** 2
      newLmbda = newLmbda / newLmbda.sum(axis=0, keepdims=1) # Normalize to make row sum=1
      newSigmaSq[d] = sigmaSq[d] / sigma
      newSigmaSq = newSigmaSq / newSigmaSq.sum(axis=0, keepdims=1) # Normalize to make row sum=1
      counter += 1
      # Check if gamma and phi converged
      if np.linalg.norm(p - oldPhi) < 1e-3 and np.linalg.norm(eta[d,:] - oldEta) < 1e-3:
        convergence[d] = 1
        phi[d] = p
        print('Document ' + str(d) + ' needed ' + str(counter) + ' iterations to converge.')
        likelihood += calcLikelihood(phi[d], eta[d,:], gamma[d,:], epsilon, reviewList[d], vocabDict, k)

  for d in range(0, M):
    newMu += newLmbda[d]
  mu = mu / M
  for d in range(0,M):
    newSigma += (newLmbda[d] - newMu)**2 + newSigmaSq[d]**2
  newSigma = newSigma / M

  return phi, eta, newMu, newSigma, likelihood

def MStep(phi, eta, reviewList, vocabDict, k, M):
  print('M-step')
  V = len(vocabDict)
  epsilon = np.zeros([k, V])
  for d in range(0, M):
    words = reviewList[d]
    for i in range(0, k):
      p = phi[d][:,i]
      for j in range(0, V):
        word = V[j]
        indicator = words.index(word)
        epsilon[i,j] += np.dot(indicator, p)
  return np.transpose(np.transpose(epsilon) / np.sum(epsilon, axis=1)) # the epsilon value

def EM(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewList, vocabDict, M,k):
  likelihood, oldLikelihood, iteration = 0, 0, 1
  while iteration <= 2 or np.abs((likelihood - oldLikelihood) / oldLikelihood) > 1e-4: # Update parameters
    oldLikelihood, oldPhi, oldEta, oldGamma, oldEpsilon, oldLambda, oldSigmaSq, oldMu, oldSigma = likelihood, phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma
    phi, eta, likelihood, mu, sigma, lmbda, sigmaSq = EStep(oldPhi, oldEta, oldGamma, oldEpsilon, oldLambda, oldSigmaSq, oldMu, oldSigma, reviewList, vocabDict, k, M)
    epsilon = MStep(phi, eta, reviewList, vocabDict, k, M)
    print('Iteration ' + str(iteration) + ': Likelihood = ' + str(likelihood))
    iteration += 1
    if iteration > 100:
      break
  return phi, eta, gamma, epsilon, mu, sigma, likelihood

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
