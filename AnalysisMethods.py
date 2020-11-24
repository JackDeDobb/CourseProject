# Use Python3
from AnalysisMethods import *
from scipy.special import digamma, gammaln
import json
import nltk
import numpy as np
import os
import random
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



def parseWordsForSentence(content, vocab, vocabDict): # Use nltk and stopwords to tokenize words for  each review
  tokenizedWords = []
  for sentence in nltk.sent_tokenize(content):
    stemmedWordsInVocab = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if stemmer.stem(w.lower()) in vocabDict]
    tokenizedWords += [vocabDict.get(w) for w in stemmedWordsInVocab]
  return tokenizedWords


def parseWords(content, stopWords): # Use nltk and stopwords to tokenize words
  tokenizedWords = []
  for sentence in nltk.sent_tokenize(content):
    stemmedWords = [stemmer.stem(w.lower()) for w in nltk.word_tokenize(sentence) if w not in string.punctuation]
    tokenizedWords += [v for v in stemmedWords if v not in stopWords] # Remove stopwords
  return tokenizedWords


def genStopwords():
  currDirectoryOfScript = os.path.dirname(os.path.realpath(__file__))
  with open('/'.join([currDirectoryOfScript, 'StopWords.json'])) as stopWords:
    return set(json.load(stopWords))


def initializeParameters(reviewList, vocabDict, M, k):
  phi, lmbda, sigmaSq = [], [], []
  eta = np.zeros([M, k])
  gamma = np.ones([M, k])
  for m in range(0, M):
    wordsInDoc = reviewList[m]
    N = len(wordsInDoc)
    phi_temp = np.ones([N, k]) * 1 / float(k)
    for i in range(0, k):
      eta[m, i] = gamma[m, i] + N / float(k)
    phi.append(phi_temp)
    lmbda.append(phi_temp)
    sigmaSq.append(np.random.rand())
    m += 1
  epsilon = np.zeros([k, len(vocabDict)])
  for i in range(0, k):
    tmp = np.random.uniform(0, 1, len(vocabDict))
    epsilon[i,:] = tmp / np.sum(tmp)
  return phi, eta, gamma, epsilon, lmbda,sigmaSq


def calcLikelihood(phi, eta, gamma, epsilon, review, vocabDict, k):
  V = len(vocabDict)
  N = len(review)
  likelihood, gammaSum, phiEtaSum, phiLogEpsilonSum, entropySum, etaSum = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0

  gammaSum += gammaln(np.sum(alpha)) # TODO: there is no alpha
  etaSum -= gammaln(np.sum(gamma))
  for i in range(0, k):
    gammaSum += -gammaln(gamma[i]) + (gamma[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta)))
    for n in range(0, N):
      if phi[n, i] > 0:
        indicator = np.sum(np.in1d(len(vocabDict), review[n]))
        phiEtaSum += phi[n, i] * (digamma(eta[i]) - digamma(np.sum(eta[:])))
        entropySum += phi[n, i] * np.log(phi[n, i])
        for j in range(0, V):
          if Beta[i,j] > 0: # TODO: there is no beta
            phiLogEpsilonSum += phi[n, i] * indicator * np.log(epsilon[i, j])
    etaSum += gammaln(eta[i]) - (eta[i] - 1) * (digamma(eta[i]) - digamma(np.sum(eta[:])))

  likelihood += (gammaSum + phiEtaSum + phiLogEpsilonSum - etaSum - entropySum) # TODO: should this be inside for loop?
  return likelihood


def EStep(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewList, vocabDict, k, M):
  print('E-step')
  newLmbda, newSigmaSq = [], []
  likelihood, newMu, newSigma = 0.0, 0.0, 0.0
  convergence = np.zeros(M)
  for d in range(0, M):
    words = reviewList[d]
    N = len(words)
    p = phi[d]
    counter = 0
    while convergence[d] == 0 and d < len(convergence):
      oldPhi = p
      p = np.zeros([N,k])
      oldEta = eta[d,:]
      for n in range(0, N):
        vocabIdx = list(vocabDict).index(words[n])
        if len(vocabIdx[0]) > 0:
          for i in range(0, k):
            e = epsilon[i, vocabIdx]
            p[n, i] = e[0][0] * np.exp(digamma(eta[d, i]) - digamma(np.sum(eta[d,:])))
          p[n,:] = p[n,:] / np.sum(p[n,:])
      eta[d,:] = gamma[d,:] + np.sum(p, axis=0)
      newLmbda[d] = 0.5 * (lmbda[d] - mu)**2
      newLmbda = newLmbda / newLmbda.sum(axis=0, keepdims=1) # Normalize to make row sum=1
      newSigmaSq[d] = sigmaSq[d] / sigma
      newSigmaSq = newSigmaSq / newSigmaSq.sum(axis=0, keepdims=1) # Normalize to make row sum=1
      counter += 1
      if np.linalg.norm(p - oldPhi) < 1e-3 and np.linalg.norm(eta[d,:] - oldEta) < 1e-3: # Check if gamma and phi converged
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
      p = phi[d][:, i]
      for j in range(0, V):
        word = V[j]
        indicator = words.index(word)
        epsilon[i,j] += np.dot(indicator, p)
  return np.transpose(np.transpose(epsilon) / np.sum(epsilon, axis=1)) # the epsilon value


def EM(phi, eta, gamma, epsilon, lmbda, sigmaSq, mu, sigma, reviewList, vocabDict, M, k):
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


