## Usage
For this program all that is required is invoking main.py using python3 and perplexity calculations will be completed accordingly.
When invoking main.py you will be prompted to enter either (A) for additive smoothing or (L) for linear smoothing which will change how perplexity for Unigrams, Bigrams, and Trigrams is aquired.

For Additive Smoothing it is possible to change the alpha values for additive smoothing which are used in the bigramAdditiveSmoothing and trigramAdditiveSmoothing() which can be seen in operation below:
'''python
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
'''
To change these values for the given data sets within the forma() functions outlined above, Simply change the observed (0.70) value to desired alpha to observe results.

For Linear Smoothing you can similarly change the value of different Lambda sets used in the output of data which can be seen below:
'''python
print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .2, .3])))
print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(train, trigram, bigram, unigram, [.3, .4, .5])))
print("Train Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(train, trigram, bigram, unigram, [.1, .3, .6])))

print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.1, .2, .3, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .2, .3])))

print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.3, .4, .5, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.3, .4, .5])))

print("Dev Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}\n".format(.1, .3, .6, trigramLinearSmoothing(dev, trigram, bigram, unigram, [.1, .3, .6])))

print("Test Linear Smoothing Trigram Perplexity using L1: {}, L2: {}, L3: {}: {}".format(.4, .5, .6, trigramLinearSmoothing(test, trigram, bigram, unigram, [.4, .5, .6])))
'''

In the above print functions functionality of the trigramLinearSmoothing function can be observed. To modify the output of the print function simply change the [?, ?, ?] lambda sets found within the call of the trigramLinearSmoothing function (Also be sure to change the ?, ?, ? found at the beginning of the format call). 
