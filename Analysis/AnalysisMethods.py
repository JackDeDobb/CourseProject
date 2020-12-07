# Use Python3
from scipy.special import digamma, gammaln
import json
import nltk
import numpy as np
import os
import ssl
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



def genStopwords():
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, '..', 'Data', 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))


def getData(folder):
  reviewDataList, itemList = [], []
  for file in os.listdir(folder):
    if file.endswith('.json'):
      with open(folder + '/' + file, encoding='utf-8') as data_file:
        reviewDataList.append(json.load(data_file))
        itemList.append(file.split('.')[0])
  return itemList, reviewDataList


def parseWords(content, stopWords): # Use nltk and stopwords to tokenize words
  tokenizedWords = []
  for sentence in nltk.sent_tokenize(content):
    stemmedWords = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if w not in string.punctuation]
    tokenizedWords += [v for v in stemmedWords if v not in stopWords] # Remove stopwords
  return tokenizedWords


def initializeParameters(reviewFreqDictList, vocabDict, M, k):
  phi, lmbda, sigmaSq = [], np.zeros(shape=(1, M)), np.zeros(shape=(1, M))
  eta = np.zeros([M, k])
  gamma = np.ones([M, k])
  for m in range(0, M):
    wordsInDoc = list(reviewFreqDictList[m].keys())
    N = len(wordsInDoc)
    phi_temp = np.ones([N, k]) * 1 / float(k)
    for i in range(0, k):
      eta[m, i] = gamma[m, i] + N / float(k)
    phi.append(phi_temp)
    lmbda[0, m] = np.random.rand()
    sigmaSq[0, m] = np.random.rand()
  lmbda = lmbda / lmbda.sum(axis=1, keepdims=1) # Normalize to make row sum=1
  sigmaSq = sigmaSq / sigmaSq.sum(axis=1, keepdims=1) # Normalize to make row sum=1
  epsilon = np.zeros([k, len(vocabDict)])
  for i in range(0, k):
    tmp = np.random.uniform(0, 1, len(vocabDict))
    epsilon[i,:] = tmp / np.sum(tmp)
  return phi, eta, gamma, epsilon, lmbda,sigmaSq


def calcLikelihood(phi, eta, gamma, epsilon, reviewDict, vocabDict, k):
  V = len(vocabDict)
  review = list(reviewDict.keys())
  N = len(review)
  gammaSum, phiEtaSum, phiLogEpsilonSum, entropySum, etaSum = 0.0, 0.0, 0.0, 0.0, 0.0

  gammaSum += gammaln(np.sum(gamma))
  etaSum -= gammaln(np.sum(eta))
  for i in range(0, k):
    gammaSum += -gammaln(gamma[i]) + (gamma[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta)))
    for n in range(0, N):
      if phi[n, i] > 0:
        indicator = np.sum(np.in1d(len(vocabDict), review[n]))
        phiEtaSum += phi[n, i] * (digamma(eta[i]) - digamma(np.sum(eta[:])))
        entropySum += phi[n, i] * np.log(phi[n, i])
        for j in range(0, V):
          if epsilon[i,j] > 0:
            phiLogEpsilonSum += phi[n, i] * indicator * np.log(epsilon[i, j])
    etaSum += gammaln(eta[i]) - (eta[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta[:])))

  return (gammaSum + phiEtaSum + phiLogEpsilonSum - etaSum - entropySum) # likelihood


def EStep(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewFreqDictList, vocabDict, k, M):
  print('E-step')
  newLmbda, newSigmaSq = np.zeros(shape=(1, M)), np.zeros(shape=(1, M))
  likelihood, newMu, newSigma = 0.0, 0.0, 0.0
  convergence = np.zeros(M)
  for d in range(0, M):
    words = list(reviewFreqDictList[d].keys())
    N = len(words)
    p = phi[d]
    counter = 0
    while convergence[d] == 0 and d < len(convergence):
      oldPhi = p
      p = np.zeros([N,k])
      oldEta = eta[d,:]
      for n in range(0, N):
        if words[n] in list(vocabDict): # If word exists in dictionary
          vocabIdx = list(vocabDict).index(words[n])
          for i in range(0, k):
            e = epsilon[i, vocabIdx]
            p[n, i] = e * np.exp(digamma(eta[d, i]) - digamma(np.sum(eta[d,:])))
          p[n,:] = p[n,:] / np.sum(p[n,:])
      eta[d,:] = gamma[d,:] + np.sum(p, axis=0)
      newLmbda[0, d] = 0.5 * (lmbda[0, d] - mu)**2
      newLmbda = newLmbda / newLmbda.sum(axis=1, keepdims=1) # Normalize to make row sum=1
      newSigmaSq[0, d] = sigmaSq[0, d] / sigma
      newSigmaSq = newSigmaSq / newSigmaSq.sum(axis=1, keepdims=1) # Normalize to make row sum=1
      counter += 1
      if np.linalg.norm(p - oldPhi) < 1e-3 and np.linalg.norm(eta[d,:] - oldEta) < 1e-3: # Check if gamma and phi converged
        convergence[d] = 1
        phi[d] = p
        print('Document ' + str(d) + ' needed ' + str(counter) + ' iterations to converge.')
        likelihood += calcLikelihood(phi[d], eta[d,:], gamma[d,:], epsilon, reviewFreqDictList[d], vocabDict, k)

  for d in range(0, M):
    newMu += newLmbda[0,d]
  mu = mu / M
  for d in range(0,M):
    newSigma += (newLmbda[0,d] - newMu)**2 + newSigmaSq[0,d]**2
  newSigma = newSigma / M

  return phi, eta, newMu, newSigma, likelihood


def MStep(phi, eta, reviewFreqDictList, vocabDict, k, M):
  print('M-step')
  V = len(vocabDict)
  epsilon = np.zeros([k, V])
  for d in range(0, M):
    words = list(reviewFreqDictList[d].keys())
    for i in range(0, k):
      p = phi[d][:, i]
      for j in range(0, V):
        word = list(vocabDict)[j]
        indicator = np.in1d(words, word).astype(int)
        epsilon[i,j] += np.dot(indicator, p)
  return np.transpose(np.transpose(epsilon) / np.sum(epsilon, axis=1)) # the epsilon value


def EM(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewFreqDictList, vocabDict, M, k):
  likelihood, oldLikelihood, iteration = 0, 0, 1
  while iteration <= 2 and (iteration <= 2 or np.abs((likelihood - oldLikelihood) / oldLikelihood) > 1e-4): # TODO: Change from 2 to 100
    oldLikelihood, oldPhi, oldEta, oldGamma, oldEpsilon, oldLambda, oldSigmaSq, oldMu, oldSigma = likelihood, phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma
    phi, eta,  mu, sigma, likelihood = EStep(oldPhi, oldEta, oldGamma, oldEpsilon, oldLambda, oldSigmaSq, oldMu, oldSigma, reviewFreqDictList, vocabDict, k, M)
    epsilon = MStep(phi, eta, reviewFreqDictList, vocabDict, k, M)
    print('Iteration ' + str(iteration) + ': Likelihood = ' + str(likelihood))
    iteration += 1
  return phi, eta, gamma, epsilon, mu, sigma, likelihood
  ## TODO : Check up on EM (it is growing on the negative side)


def sentenceLabeling(mu, sigma, reviewFreqDictList, shapeSize): # Update labels
  reviewLabelList = [[] for i in range(len(reviewFreqDictList))]
  for i in range(len(reviewFreqDictList)):
    aspectWeights = np.zeros(shape=(shapeSize, len(list(reviewFreqDictList[i].keys()))))
    for j in range(shapeSize):
      aspectWeights[j] = np.random.normal(loc=mu, scale=sigma, size=len(list(reviewFreqDictList[i].keys())))
    aspectWeights = aspectWeights / aspectWeights.sum(axis=1, keepdims=1) # Normalize to make row sum=1
    for j in range(shapeSize):
      reviewLabels = [-1] * len(list(reviewFreqDictList[i].keys())) # Initialize each review as -1
      reviewLabels[np.where(aspectWeights[j] == max(aspectWeights[j]))[0][0]] = 1 # Change the label to 1 for the word most matching the aspec
      reviewLabelList[i].append(reviewLabels)
  return reviewLabelList


def generateAspectParameters(reviewFreqDictList, vocabDict): # Aspect modeling
  k = 4 # nbr of latent states z
  M = len(reviewFreqDictList) # nbr of reviews
  initMu, initSigma = 0.0, 0.0
  initPhi, initEta, initGamma, initEpsilon, initLambda, initSigmaSq = initializeParameters(reviewFreqDictList, vocabDict, M, k)
  for d in range(0, M):
    initMu += initLambda[0,d]
  initMu = initMu / M
  for d in range(0, M):
    initSigma += (initLambda[0, d] - initMu)**2 + initSigmaSq[0,d]**2
  initSigma = initSigma / M
  phi, eta, gamma, epsilon, mu, sigma, likelihood = EM(initPhi, initEta, initGamma, initEpsilon, initLambda, initSigmaSq, initMu, initSigma, reviewFreqDictList, vocabDict, M, k)
  return mu, sigma


def getStats(predList, allReviewsList):
  totalMSE = np.square(np.subtract(predList, allReviewsList)).mean()
  totalPearson = np.corrcoef(predList, allReviewsList)[0, 1]
  return totalMSE, totalPearson


def generateResults(idList, reviewIdList, reviewContentList, reviewRatingList, reviewAuthorList, reviewDataList, reviewLabelList, reviewList, reviewMatrixList, positiveWordList, negativeWordList, totalMse, totalPearson, finalFile):
  f = open(finalFile, 'w')
  for i in range(len(reviewList)):
     f.write(':'.join([idList[i], reviewIdList[i], reviewContentList[i], str(reviewList[i]), str(reviewMatrixList[i])]) + '\n')
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
  print("Total number of items =" + str(len(set(idList))) +"\n")
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