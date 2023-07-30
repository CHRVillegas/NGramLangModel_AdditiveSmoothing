'''
    Christopher Villegas
    CSE 143 Assignment 2
    May 18 2023
'''

import math
if __name__ == "__main__":
    """
    def unigramData():
        Gets word counter per sentence in the data
        Adds <start> and <end> tokens to the end of each individual sentence
        Breaks resulting sentences into smaller individual word tokens
    """
    def unigramData(wordCount, remWords):
        train = open("A2-Data/1b_benchmark.train.tokens", "r", encoding="utf8")
        for s in train:
            tokens = s.split()
            tokens.insert(0, "<start>")
            tokens.append("<end>")

            for i in tokens:
                wordCount[i] = 1 + wordCount.get(i, 0)
        
        train.close()
        wordCount["UNK"] = 0
        for i, c in wordCount.items():
            if c < 3:
                remWords.append(i)
                wordCount["UNK"] += c
        
        for word in remWords:
            del wordCount[word]

    """
    def MLE():
        Counts total amount of <start> tokens
        Calculates MLE if token is found that is not <start>
    """
    def MLE(wordCount, prob):

        sumStart = wordCount["<start>"]
        sumNotStart = sum(wordCount.values()) - sumStart

        for i, c in wordCount.items():
            if i == "<start>":
                continue
            prob[i] = c / sumNotStart

    """
    def additiveSmoothing():
        Performs MLE with additive smoothing
        Aquires sum of <start> tokens
        Calculates MLE if token seen is not <start>
    """
    def additiveSmoothing(wordCount, prob, alpha):
        sumStart = wordCount["<start>"]
        sumNotStart = sum(wordCount.values()) - sumStart
        for i, c in wordCount.items():
            if i == "<start>":
                continue
            prob[i] = (c+alpha) / (c+alpha * sumNotStart)

    """
    def bigramAdditiveSmoothing():
        Gets bigram probability with additive smoothing
        Breaks up sentence into individual word tokens
        For each element and following element in the token list add them to bigram dictionary
        Get bigram probability using bigramProb function
        Calculate perplexity and return
    """
    def bigramAdditiveSmoothing(data, mleprob, unigramprob, alpha):
        sentSum, totSum, senLen = 0, 0, 0

        for s in data:
            tokens = s.split()
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            for bigram in bigrams:
                if (bigram[0] == "<start>" and bigram[1] == "<start>"):
                    continue
                if bigramProbability(bigram, mleprob, unigramprob) != 0:
                    sentSum += math.log(bigramProbSmooth(bigram, mleprob, unigramprob, 1))
            
            totSum += sentSum
            sentSum = 0
            senLen += len(tokens)-2

        inverse = float(-1) / float(senLen)
        exponent = inverse * (totSum)
        expon = math.exp(exponent)
        return expon
    
    """
    def bigramProbSmooth():
        MLE for bigram with additive smoothing
        Get probability of bigram and calculate MLE
        Return 0 otherwise
    """
    def bigramProbSmooth(featureCount, mleprob, unigramprob, alpha):
        try:
            bigramprob = mleprob[featureCount]
            sumwordCount = unigramprob[featureCount[0]]

            return float(bigramprob + alpha) / float(bigramprob + alpha * sumwordCount)
        
        except:
            return 0
        
    """
    def bigramProbability():
        MLE for bigram without additive smoothing
        Get probability of bigram and calculate MLE
        Return 0 otherwise
    """
    def bigramProbability(featureCount, mleprob, unigramprob):
        try:
            bigramprob = mleprob[featureCount]
            sumWordCount = unigramprob[featureCount[0]]

            return float(bigramprob) / float(sumWordCount)
        except:
            return 0

    """
    def bigramPer():
        Calculate bigram perplexity
        Break up each sentence into individual word tokens
        For each bigram calculate total sentence length 
        Calculate perplexity and return
    """    
    def bigramPer(data, mleprob, unigramprob):
        sentSum, totSum, senLen = 0, 0, 0
        for s in data:
            tokens = s.split()
            bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            for bigram in bigrams:
                if (bigram[0] == "<start>" and bigram[1] == "<start>"):
                    continue
                if bigramProbability(bigram, mleprob, unigramprob) != 0:
                    sentSum += math.log(bigramProbability(bigram, mleprob, unigramprob))
            
            totSum += sentSum
            sentSum = 0
            senLen+= len(tokens) - 2

        inverse = float(-1) / float(senLen)
        exponent = inverse * (totSum)
        expon = math.exp(exponent)
        return expon
    
    '''
    def unigramPer():
        Calculate unigram perplexity
        Break up each sentence into individual word tokens
        For each unigram calculate total sentence length 
        Calculate perplexity and return
    '''
    def unigramPer(data, mleprob):
        sentSum, totSum, senLen = 0, 0, 0
        for s in data:
            tokens = s.split()
            for unigram in tokens:
                if unigram == "<start>":
                    continue
                sentSum += math.log(mleprob[unigram])
                senLen+=1

            totSum += sentSum
            sentSum = 0
        inverse = float(-1) / float(senLen)
        exponent = inverse * (totSum)
        expon = math.exp(exponent)
        return expon
    
    '''
    def trigramProbabilitySmoothing():
        If the desired key for trigram in dictionary exists
        Get probability of the trigram
        Return its probability with additive smoothing
        Else return 1
    '''
    def trigramProbabilitySmoothing(trigram, mleprob, unigramprob, wordCount, alpha):
        try:
            trigramprob = mleprob[trigram]
            if trigram[0] == "<start>" and trigram[1] == "<start>":
                sumwordCount = wordCount["<start>"]
            else:
                sumwordCount = unigramprob[trigram[0:2]]

            return float(trigramprob + alpha) / float(trigramprob+alpha*sumwordCount)
        except:
            return 1
        

    '''
    def trigramProbability():
        If the desired key for trigram in dictionary exists
        Get prob of trigram
        Return its prob without additive smoothing
        Else return 1
    '''
    def trigramProbability(trigram, mleprob, unigramprob, wordCount):
        try:
            trigramprob = mleprob[trigram]
            if trigram[0] == "<start>" and trigram[1] == "<start>":
                sumwordCount = wordCount["<start>"]
            else:
                sumwordCount = unigramprob[trigram[0:2]]

            return float(trigramprob) / float(sumwordCount)
        
        except:
            return 1
    
    '''
    def trigramAdditiveSmoothing():
        Get Trigram Probability with additive smoothing
        Break up sentences into individual tokens and reconstruct into trigrams
        Calculate perplexity and Return
    '''
    def trigramAdditiveSmoothing(data, mleprob, unigramprob, wordCount, alpha):
        sentSum, totSum, senLen = 0, 0, 0
        for s in data:
            tokens = s.split()
            trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens) - 2)]
            for trigram in trigrams:
                if trigramProbability(trigram, mleprob, unigramprob, wordCount) != 0:
                    sentSum += math.log(trigramProbabilitySmoothing(trigram, mleprob, unigramprob, wordCount, alpha))

            totSum += sentSum
            sentSum = 0
            senLen += len(tokens) - 2

        inverse = float(-1) / float(senLen)
        exponent = inverse * totSum
        expon = math.exp(exponent)
        return expon
    
    '''
    def trigPerplexity():
        Calculate trigram prob and perplexity without additive smoothing
        Break up sentences into individual tokens and reconstruct into trigrams
        Calculate perplexity and Return
    '''
    def trigPerplexity(data, mleprob, unigramprob, wordCount):
        sentSum, totSum, senLen = 0, 0, 0
        for s in data:
            tokens = s.split()
            trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]
            for trigram in trigrams:
                if trigramProbability(trigram, mleprob, unigramprob, wordCount) != 0:
                    sentSum += math.log(trigramProbability(trigram, mleprob, unigramprob, wordCount))

            totSum += sentSum
            sentSum = 0
            senLen += len(tokens) - 2

        inverse = float(-1) / float(senLen)
        exponent = inverse * totSum
        expon = math.exp(exponent)
        return expon
    
    '''
    def trigramLinearSmoothing():
        Calculate trigram probability and perplexity using Linear smoothing
        Break up sentences into individual tokens and reconstruct into trigrams
        Calculate perplexity and return
    '''
    def trigramLinearSmoothing(data, mleprob, unigramprob, wordCount, l):
        sentSum, totSum, senLen = 0, 0, 0
        startCount = wordCount["<start>"]
        totTokens = sum(wordCount.values()) - startCount
        for s in data:
            tokens = s.split()
            trigrams = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]

            for trigram in trigrams:
                insideCount = 0
                insideCount += (l[0] * float(wordCount[trigram[2]] / totTokens)) + (l[1] * (bigramProbability(trigram[1:3], unigramprob, wordCount))) + (l[2] * (trigramProbability(trigram, mleprob, unigramprob, wordCount)))
                sentSum += math.log(insideCount)

            totSum += sentSum
            sentSum = 0
            senLen += len(tokens)-2

        inverse = float(-1) / float(senLen)
        exponent = inverse * totSum
        expon = math.exp(exponent)
        return expon
    
    '''
    def getData():
        Reads data from different files
        Adds <start> and <end> to different sentences
    '''
    def getData(words, sentences, doc):
        if doc == "dev":
            data = open("A2-Data/1b_benchmark.dev.tokens", "r", encoding="utf8")
        elif doc == "test":
            data = open("A2-Data/1b_benchmark.test.tokens", "r", encoding="utf8")
        else:
            data = open("A2-Data/1b_benchmark.train.tokens", encoding="utf8")

        # for each sentence in data
        for s in data:

            tokens = s.split()
  
            for word in tokens:

                if word not in words:
 
                    tokens[tokens.index(word)] = "UNK"

            tokens.insert(0, "<start>")

            tokens.insert(0, "<start>")

            tokens.append("<end>")

            sentences.append(" ".join(tokens))
        data.close()


    '''
    def bigramTrigram():
        Get Bigrams and Trigrams from data
    '''
    def bigramTrigram(data, wordCount, tokenCount):
        for s in data:
            tokens = s.split()
            if tokenCount == 2:
                ngram = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
            if tokenCount == 3:
                ngram = [(tokens[i], tokens[i+1], tokens[i+2]) for i in range(len(tokens)-2)]

            for i in ngram:
                wordCount[i] = 1+wordCount.get(i, 0)

    unigram, bigram, trigram, prob, = {}, {}, {}, {}
    unk, train, dev, test = [], [], [], []

    unigramData(unigram, unk)

    val = input("L for Linear or A for Additive: ")
    while True:
        if val == 'L' or val == 'l' or val == 'A' or val == 'a':
            break
        print("Incorrect Selection, Please Input again.")
        val = input("L for Linear or A for Additive: ")

    if val == 'A' or val == 'a':
        print("\nModel Vocabulary Unique Tokens: {}".format(len(unigram) - 1))
        print("Additive Smoothing with alpha=1:")
        getData(unigram, train, doc="train")
        additiveSmoothing(unigram, prob, 0.70)
        print("Train Unigram Perplexity: {}".format(unigramPer(train, prob)))
        getData(unigram, dev, doc="dev")
 
        print("Dev Unigram Perplexity: {}".format(unigramPer(dev, prob)))
        getData(unigram, test, doc="test")
        print("Test Unigram Perplexity: {}\n".format(unigramPer(test, prob)))

        bigramTrigram(train, bigram, 2)
        print("Model Vocabulary Unique Bigram Count: {}".format(len(bigram) - 1))
        print("Train Bigram Perplexity: {}".format(bigramAdditiveSmoothing(train, bigram, unigram, 0.70)))
        print("Dev Bigram Perplexity: {}".format(bigramAdditiveSmoothing(dev, bigram, unigram, 0.70)))
        print("Test Bigram Perplexity: {}\n".format(bigramAdditiveSmoothing(test, bigram, unigram, 0.70)))

        bigramTrigram(train, trigram, 3)
        print("Model Vocabulary Unique Trigram Count: {}".format(len(trigram) - 1))
        print("Train Trigram Perplexity: {}".format(trigramAdditiveSmoothing(train, trigram, bigram, unigram, 0.70)))
        print("Dev Trigram Perplexity: {}".format(trigramAdditiveSmoothing(dev, trigram, bigram, unigram, 0.70)))
        print("Test Trigram Perplexity: {}\n".format(trigramAdditiveSmoothing(test, trigram, bigram, unigram, 0.70)))

    else:
        print("\nModel Vocabulary Unique Tokens: {}".format(len(unigram) - 1))

        print("No Smoothing:")
        getData(unigram, train, doc="train")
        MLE(unigram, prob)
        print("Train Unigram Perplexity: {}".format(unigramPer(train, prob)))
        getData(unigram, dev, doc="dev")
   
        print("Dev Unigram Perplexity: {}".format(unigramPer(dev, prob)))
        getData(unigram, test, doc="test")
        print("Test Unigram Perplexity: {}\n".format(unigramPer(test, prob)))

        bigramTrigram(train, bigram, 2)
        print("Model Vocabulary Unique Bigram Count: {}".format(len(bigram) - 1))
        print("Train Bigram Perplexity: {}".format(bigramPer(train, bigram, unigram)))
        print("Dev Bigram Perplexity: {}".format(bigramPer(dev, bigram, unigram)))
        print("Test Bigram Perplexity: {}\n".format(bigramPer(test, bigram, unigram)))

        bigramTrigram(train, trigram, 3)
        print("Model Vocabulary Unique Trigram Count: {}".format(len(trigram) - 1))
        print("Train Trigram Perplexity: {}".format(trigPerplexity(train, trigram, bigram, unigram)))
        print("Dev Trigram Perplexity: {}".format(trigPerplexity(dev, trigram, bigram, unigram)))
        print("Test Trigram Perplexity: {}\n".format(trigPerplexity(test, trigram, bigram, unigram)))

        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .2, .3])))
        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(train, trigram, bigram, unigram, [.3, .4, .5])))
        print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .3, .6])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .2, .3])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.3, .4, .5])))

        print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .3, .6])))

        print("Test Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.4, .5, .6, trigramLinearSmoothing(test, trigram, bigram, unigram, [.4, .5, .6])))



