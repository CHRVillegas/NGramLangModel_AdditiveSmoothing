## Usage
For this program all that is required is invoking main.py using python3 and perplexity calculations will be completed accordingly.
When invoking main.py you will be prompted to enter either (A) for additive smoothing or (L) for linear smoothing which will change how perplexity for Unigrams, Bigrams, and Trigrams is aquired.

For Additive Smoothing it is possible to change the alpha values for additive smoothing which are used in the bigramAdditiveSmoothing and trigramAdditiveSmoothing().

To change these values for the given data sets within the format() functions, Simply change the observed (0.70) value to desired alpha to observe results.

For Linear Smoothing you can similarly change the value of different Lambda sets used in the output of data which can be seen below:

To modify the output of the print function simply change the [?, ?, ?] lambda sets found within the call of the trigramLinearSmoothing function (Also be sure to change the ?, ?, ? found at the beginning of the format call). 
