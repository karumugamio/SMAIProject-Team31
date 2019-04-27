'''
SMAI - CSE471 - Spring 2019
Predicting movie tagline project
'''
import logging
import re
import pickle
from collections import Counter

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords

import datautil as util

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s %(message)s')

inputDataFileName = 'data/movies_data.csv'
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
    for summary in moviesData.overview:
        c_movieSummary.append(util.clean_text(summary,remove_stopwords = False))
    debug("Summaries Cleaned up")

    #util.write_list_to_file(c_movieSummary,'summarylist.csv')
    c_movieTagline = []
    for text in moviesData.tagline:
        c_movieTagline.append(util.clean_text(text))
    debug("MovieTagline Cleaned up")
    #util.write_list_to_file(c_movieTagline,'taglinelist.csv')


    debug(type(c_movieSummary))
    debug(type(c_movieTagline))
    debug(len(c_movieSummary))
    debug(len(c_movieTagline))

    word_counts = {}

    util.count_words(word_counts, c_movieSummary)
    print("Size of Vocabulary:", len(word_counts))
    
        
    util.count_words(word_counts, c_movieTagline)
            
    print("Size of Vocabulary:", len(word_counts))

    # Load Conceptnet Numberbatch's (CN) embeddings, similar to GloVe, but probably better 
    # (https://github.com/commonsense/conceptnet-numberbatch)
    embeddings_index = {}
    with open('data/numberbatch-en.txt', encoding='utf-8') as f:
        for line in f:
            values = line.split(' ')
            word = values[0]
            embedding = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = embedding

    print('Word embeddings:', len(embeddings_index))



    debug('End of prepareData fn')
    # Find the number of words that are missing from CN, and are used more than our threshold.
    missing_words = 0
    threshold = 20

    for word, count in word_counts.items():
        if count > threshold:
            if word not in embeddings_index:
                missing_words += 1
                
    missing_ratio = round(missing_words/len(word_counts),4)*100
                
    debug("Number of words missing from CN:"+ str(missing_words))
    debug("Percent of words that are missing from vocabulary: {}%".format(missing_ratio))

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

    debug("Total number of unique words:"+str(len(word_counts)))
    debug("Number of words we will use:"+str(len(vocab_to_int)))
    debug("Percent of words we will use: {}%".format(usage_ratio))

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
    debug(len(word_embedding_matrix))

    # Apply convert_to_ints to clean_summaries and clean_texts
    word_count = 0
    unk_count = 0

    int_summaries, word_count, unk_count = util.convert_to_ints(c_movieSummary, word_count, unk_count,vocab_to_int)
    int_texts, word_count, unk_count = util.convert_to_ints(c_movieTagline, word_count, unk_count, vocab_to_int,eos=True)

    unk_percent = round(unk_count/word_count,4)*100

    debug("Total number of words in headlines:"+str(word_count))
    debug("Total number of UNKs in headlines:"+str(unk_count))
    debug("Percent of words that are UNK: {}%".format(unk_percent))
    lengths_summaries = util.create_lengths(int_summaries)
    lengths_texts = util.create_lengths(int_texts)

    debug("Summaries:")
    debug(lengths_summaries.describe())
    debug("Texts:")
    debug(lengths_texts.describe())

    # Inspect the length of texts
    debug(np.percentile(lengths_texts.counts, 90))
    debug(np.percentile(lengths_texts.counts, 95))
    debug(np.percentile(lengths_texts.counts, 99))

    # Inspect the length of summaries
    debug(np.percentile(lengths_summaries.counts, 90))
    debug(np.percentile(lengths_summaries.counts, 95))
    debug(np.percentile(lengths_summaries.counts, 99))


    sorted_summaries = []
    sorted_texts = []
    max_text_length = 84
    max_summary_length = 13
    min_length = 2
    unk_text_limit = 1
    unk_summary_limit = 0

    intforUnKnown = vocab_to_int["<UNK>"]
    for length in range(min(lengths_texts.counts), max_text_length): 
        for count, words in enumerate(int_summaries):
            if (len(int_summaries[count]) >= min_length and
                len(int_summaries[count]) <= max_summary_length and
                len(int_texts[count]) >= min_length and
                util.unk_counter(int_summaries[count],intforUnKnown) <= unk_summary_limit and
                util.unk_counter(int_texts[count],intforUnKnown) <= unk_text_limit and
                length == len(int_texts[count])
            ):
                sorted_summaries.append(int_summaries[count])
                sorted_texts.append(int_texts[count])
            
    # Compare lengths to ensure they match
    debug(len(sorted_summaries))
    debug(len(sorted_texts))

    f = open("int_to_vocab.pkl","wb")
    pickle.dump(int_to_vocab,f)
    f.close()

    f = open("vocab_to_int.pkl","wb")
    pickle.dump(vocab_to_int,f)
    f.close()
    print(type(int_to_vocab))
    print(type(vocab_to_int))
    print(type(c_movieTagline))

    np.array(c_movieTagline).dump(open('cleantxt.npy', 'wb'))
    myArray = np.load(open('cleantxt.npy', 'rb'),allow_pickle=True)

    print(type(myArray))
    np.array(word_embedding_matrix).dump(open('word_embedding_matrix.npy', 'wb'))
    np.array(sorted_summaries).dump(open('sorted_summaries.npy', 'wb'))
    np.array(sorted_texts).dump(open('sorted_texts.npy', 'wb'))


# Calling main function
if __name__ == "__main__":
	run()