'''
SMAI - CSE471 - Spring 2019
Predicting movie tagline project
'''
import logging
import re
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import datautil as util

logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')

inputDataFileName = 'data/movies_text2.csv'
def debug(arg):
    #logging.debug(arg)
    print(arg)

def info(arg):
    #logging.info(arg)
    print(arg)

def warn(arg):
    #logging.warn(arg)
    print(arg)

def run():
    #nltk.download("stopwords") - Verify how to do this effectively. - Loading stop words
    prepareData(inputDataFileName)
    debug('End of the program')

def prepareData(inputFileName):

    moviesData = pd.read_csv(inputFileName)
    debug("Size of data read from file is "+str(moviesData.shape))
    debug("\nPreview of the Data set")
    debug(moviesData.head())
    debug("\nChecking for Null Values in the data set")
    debug(moviesData.isnull().sum())

    #remove all NA elements - we cannot work with missing values in this problem
    # Remove null values and unneeded features
    moviesData = moviesData.dropna()
    moviesData = moviesData.reset_index(drop=True)
    debug("\nPreview of the Data set after Clean up of missing values")
    debug(moviesData.head())
    debug("\nChecking for Null Values in the data set")
    debug(moviesData.isnull().sum())
    debug("\nSize of data read after Clean up for the missing data"+str(moviesData.shape))
    
    c_movieSummary =[]
    for summary in moviesData.Summary:
        c_movieSummary.append(util.clean_text(summary,remove_stopwords = False))
    debug("Summaries Cleaned up")

    #util.write_list_to_file(c_movieSummary,'summarylist.csv')
    c_movieTagline = []
    for text in moviesData.Text:
        c_movieTagline.append(util.clean_text(text))
    debug("MovieTagline Cleaned up")
    #util.write_list_to_file(c_movieTagline,'taglinelist.csv')





    #moviesData['Summary'].apply(util.clean_text_inclstopwords)
    #moviesData['Text'].apply(util.clean_text)
    #moviesData.to_csv('test.csv')
    #c_movieSummary = moviesData['Summary'].values    # cleaned list of movie summary
    #c_movieTagline = moviesData['Text'].values       # cleaned list of movie tag lines

    debug(len(c_movieSummary))
    debug(len(c_movieTagline))

    def count_words(count_dict, text):
        '''Count the number of occurrences of each word in a set of text'''
        for sentence in text:
            
            for word in sentence.split():
                if word not in count_dict:
                    count_dict[word] = 1
                else:
                    count_dict[word] += 1
    word_counts = {}

    count_words(word_counts, c_movieSummary)
    print("Size of Vocabulary:", len(word_counts))
    
        
    count_words(word_counts, c_movieTagline)
            
    print("Size of Vocabulary:", len(word_counts))


    with open('test2.csv', 'w') as ff:
        for key in word_counts.keys():
            ff.write("%s,%s\n"%(key,word_counts[key]))
    
    ff.close()

    wordFrequencyDetails = {}

    wordFrequencyDetails.update(util.count_words(wordFrequencyDetails,c_movieSummary))
    c = Counter(c_movieSummary)  
    debug("Size of Vocabulary:"+str(len(wordFrequencyDetails)))
    debug("Size of Vocabulary:"+str(len(c)))

    wordFrequencyDetails.update(util.count_words(wordFrequencyDetails,c_movieTagline))

    '''with open('test2.csv', 'w') as ff:
        for key in wordFrequencyDetails.keys():
                ff.write("%s,%s\n"%(key,wordFrequencyDetails[key]))'''

                              

    debug("Size of Vocabulary:"+str(len(wordFrequencyDetails)))



    debug('End of prepareData fn')



# Calling main function
if __name__ == "__main__":
	run()
'''
reviews = pd.read_csv("movies_text.csv")

# Remove null values and unneeded features
reviews = reviews.dropna()
#reviews = reviews.drop(['Id','ProductId','UserId','ProfileName','HelpfulnessNumerator','HelpfulnessDenominator',
                        #'Score','Time'], 1)
reviews = reviews.reset_index(drop=True)
# Clean the summaries and texts
clean_summaries = []
for summary in reviews.Summary:
    clean_summaries.append(clean_text(summary, remove_stopwords=False))
print("Summaries are complete.")

clean_texts = []
for text in reviews.Text:
    clean_texts.append(clean_text(text))
print("Texts are complete.")

# Find the number of times each word was used and the size of the vocabulary
word_counts = {}

count_words(word_counts, clean_summaries)
count_words(word_counts, clean_texts)
            
print("Size of Vocabulary:", len(word_counts))

# Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
# (https://github.com/commonsense/conceptnet-numberbatch)
embeddings_index = {}
with open('gdrive/My Drive/numberbatch-en.txt', encoding='utf-8') as f:
    for line in f:
        values = line.split(' ')
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = embedding

print('Word embeddings:', len(embeddings_index))

# Find the number of words that are missing from CN, and are used more than our threshold.
missing_words = 0
threshold = 20

for word, count in word_counts.items():
    if count > threshold:
        if word not in embeddings_index:
            missing_words += 1
            
missing_ratio = round(missing_words/len(word_counts),4)*100
            
print("Number of words missing from CN:", missing_words)
print("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

# Limit the vocab that we will use to words that appear ≥ threshold or are in GloVe

#dictionary to convert words to integers
vocab_to_int = {} 

value = 0
for word, count in word_counts.items():
    if count >= threshold or word in embeddings_index:
        vocab_to_int[word] = value
        value += 1

# Special tokens that will be added to our vocab
codes = ["<UNK>","<PAD>","<EOS>","<GO>"]   

# Add codes to vocab
for code in codes:
    vocab_to_int[code] = len(vocab_to_int)

# Dictionary to convert integers to words
int_to_vocab = {}
for word, value in vocab_to_int.items():
    int_to_vocab[value] = word

usage_ratio = round(len(vocab_to_int) / len(word_counts),4)*100

print("Total number of unique words:", len(word_counts))
print("Number of words we will use:", len(vocab_to_int))
print("Percent of words we will use: {}%".format(usage_ratio))

# Need to use 300 for embedding dimensions to match CN's vectors.
embedding_dim = 300
nb_words = len(vocab_to_int)

# Create matrix with default values of zero
word_embedding_matrix = np.zeros((nb_words, embedding_dim), dtype=np.float32)
for word, i in vocab_to_int.items():
    if word in embeddings_index:
        word_embedding_matrix[i] = embeddings_index[word]
    else:
        # If word not in CN, create a random embedding for it
        new_embedding = np.array(np.random.uniform(-1.0, 1.0, embedding_dim))
        embeddings_index[word] = new_embedding
        word_embedding_matrix[i] = new_embedding

# Check if value matches len(vocab_to_int)
print(len(word_embedding_matrix))
'''