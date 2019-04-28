import logging
import math
import os
import pickle
import re
import time
import warnings

import nltk
import numpy as np
import pandas as pd
import tensorflow as tf
from nltk.corpus import stopwords

import RNNHelper as rnnhelper
import datautil as util


CHECKPOINT_BEST_MODEL_FILE = 'best_model.ckpt'
CHECKPOINT_METAFILE = ''

INT_TO_VOCAB_FLNAME = 'int_to_vocab.pkl'
VOCAB_TO_INT_FLNAME = 'vocab_to_int.pkl'
CLEAN_TXT_FLNAME = 'cleantxt.npy'
WORD_EMBEDDING_FLNAME = 'word_embedding_matrix.npy'
SYNOPSIS_FLNAME = 'sorted_synopsis.npy'
TAGLINE_FLNAME = 'sorted_tagline.npy'


logging.basicConfig(level=logging.DEBUG,format='%(asctime)s %(levelname)s %(message)s')

def debug(arg):
    logging.debug(arg)

def info(arg):
    logging.info(arg)

def warn(arg):
    logging.warn(arg)

class RNNSeq2SeqModel:
    def __init__(self):
        self.epochs = 100
        self.batch_size = 64
        self.rnn_size = 256
        self.num_layers = 2
        self.learning_rate = 0.005
        self.keep_probability = 0.75
        self.vocab_to_int = {}
        self.word_embedding_matrix=[]
        self.int_to_vocab = {}
        self.clean_texts = {}
        self.sorted_synopsis = []
        self.sorted_tagline = []
        self.sorted_synopsis_validation = []
        self.sorted_tagline_validation = []
        self.sorted_synopsis_test = []
        self.sorted_tagline_test = []

        # Parameters related to Train the Model
        self.learning_rate_decay = 0.95
        self.min_learning_rate = 0.0005
        self.display_step = 5 # Check training loss after every 20 batches
        self.stop_early = 0 
        self.stop = 3 # If the update loss does not decrease in 3 consecutive update checks, stop training
        self.per_epoch = 3 # Make 3 update checks per epoch
        self.update_check = 0
        self.update_loss = 0 
        self.batch_loss = 0

        debug("Initialized RNN Model Object")

    def fit(self):
        debug("Fit the model")

        warnings.filterwarnings("ignore")

        # Build the graph
        train_graph = tf.Graph()
        # Set the graph to default to ensure that it is ready for training

        with train_graph.as_default():
            # Load the model inputs    
            input_data, targets, lr, keep_prob, summary_length, max_summary_length, text_length = rnnhelper.model_inputs()

            # Create the training and inference logits
            training_logits, inference_logits = rnnhelper.seq2seq_model(tf.reverse(input_data, [-1]),
                                                            targets, 
                                                            keep_prob,   
                                                            text_length,
                                                            summary_length,
                                                            max_summary_length,
                                                            len(self.vocab_to_int)+1,
                                                            self.rnn_size, 
                                                            self.num_layers, 
                                                            self.vocab_to_int,
                                                            self.batch_size,
                                                            self.word_embedding_matrix)
            # Create tensors for the training logits and inference logits
            training_logits = tf.identity(training_logits.rnn_output, 'logits')
            inference_logits = tf.identity(inference_logits.sample_id, name='predictions')
            
            # Create the weights for sequence_loss
            masks = tf.sequence_mask(summary_length, max_summary_length, dtype=tf.float32, name='masks')
        
            with tf.name_scope("optimization"):
                # Loss function
                cost = tf.contrib.seq2seq.sequence_loss(training_logits,targets,masks)

                # Optimizer
                optimizer = tf.train.AdamOptimizer(self.learning_rate)
                # Gradient Clipping
                gradients = optimizer.compute_gradients(cost)
                capped_gradients = [(tf.clip_by_value(grad, -5., 5.), var) for grad, var in gradients if grad is not None]
                train_op = optimizer.apply_gradients(capped_gradients)

        debug("Graph is built.")
        debug("Size of Training Data.")
        sorted_synopsis_train = self.sorted_synopsis
        sorted_tagline_train = self.sorted_tagline
        debug(len(sorted_synopsis_train))
        debug(len(sorted_tagline_train))

        self.update_check = (len(sorted_tagline_train)//self.batch_size//self.per_epoch)-1

        summary_update_loss = []
        checkpoint = CHECKPOINT_BEST_MODEL_FILE
        
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If we want to continue training a previous session
            #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            #loader.restore(sess, checkpoint)
            for epoch_i in range(1, self.epochs+1):
                update_loss = 0
                batch_loss = 0 
                intValueForPad = self.vocab_to_int['<PAD>']    
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(rnnhelper.get_batches(sorted_synopsis_train, sorted_tagline_train, self.batch_size,intValueForPad)):
                    debug('Training Epoch #['+str(epoch_i)+'] Batch number ['+str(batch_i)+']')
                    start_time = time.time()
                    _, loss = sess.run([train_op, cost],{input_data: texts_batch,targets: summaries_batch,lr: self.learning_rate,summary_length: summaries_lengths,text_length: texts_lengths,keep_prob: self.keep_probability})          
                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    if batch_i % self.display_step == 0 and batch_i > 0:
                        debug('Training Epoch #['+str(epoch_i)+'] Batch number ['+str(batch_i)+']')
                        debug('Loss #['+str(epoch_i)+'] Seconds ['+str(batch_i)+']')
                        debug('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'.format(epoch_i,self.epochs,batch_i,len(sorted_tagline_train) // self.batch_size, batch_loss / self.display_step,batch_time*self.display_step))
                        batch_loss = 0
                    if batch_i % self.update_check == 0 and batch_i > 0:
                        debug("Average loss for this update:"+str(round(self.update_loss/self.update_check,3)))
                        summary_update_loss.append(self.update_loss)

                        # If the update loss is at a new minimum, save the model
                        if self.update_loss <= min(summary_update_loss):
                            debug('New Record!') 
                            self.stop_early = 0
                            saver = tf.train.Saver() 
                            saver.save(sess, checkpoint)
                        else:
                            debug("No Improvement.")
                            self.stop_early += 1
                            if self.stop_early == self.stop:
                                break
                        self.update_loss = 0

                # Reduce learning rate, but not below its minimum value
                self.learning_rate *= self.learning_rate_decay
                if self.learning_rate < self.min_learning_rate:
                    self.learning_rate = self.min_learning_rate
        
                if self.stop_early == self.stop:
                    debug("Stopping Training.")
                    break
        debug("Training Completed")


    def predict(self,input_sentence):
        
        text = self.text_to_seq(input_sentence)
        #random = np.random.randint(0,len(clean_texts))
        #input_sentence = clean_texts[random]
        #text = text_to_seq(clean_texts[random])

        checkpoint = CHECKPOINT_BEST_MODEL_FILE

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            predObj = loaded_graph.get_tensor_by_name('predictions:0')
            text_length = loaded_graph.get_tensor_by_name('text_length:0')
            summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
            #Multiply by batch_size to match the model's input parameters
            predictionsValue = sess.run(predObj, {input_data: [text]*self.batch_size, 
                                      summary_length: [np.random.randint(5,8)], 
                                      text_length: [len(text)]*self.batch_size,
                                      keep_prob: 1.0})[0] 

        # Remove the padding from the tweet
        pad = self.vocab_to_int["<PAD>"] 

        debug('Original Text:['+str(input_sentence)+']')

        debug('Text')
        debug('  Word Ids:    {}'.format([i for i in text]))
        debug('  Input Words: {}'.format(" ".join([self.int_to_vocab[i] for i in text])))

        debug('Summary')
        debug('  Word Ids:       {}'.format([i for i in predictionsValue if i != pad]))
        debug('  Response Words: {}'.format(" ".join([self.int_to_vocab[i] for i in predictionsValue if i != pad])))
        debug("predict values")
        returnPredict =  " ".join([self.int_to_vocab[i] for i in predictionsValue if i != pad])
        return returnPredict
    
    def getGenerateTagline(self,predictionsValue):
        pad = self.vocab_to_int["<PAD>"] 
        returnPredict =  " ".join([self.int_to_vocab[i] for i in predictionsValue if i != pad])
        return returnPredict 


    def predictFromProcessedSentence(self,text):
        checkpoint = CHECKPOINT_BEST_MODEL_FILE

        loaded_graph = tf.Graph()
        with tf.Session(graph=loaded_graph) as sess:
            # Load saved model
            loader = tf.train.import_meta_graph(checkpoint + '.meta')
            loader.restore(sess, checkpoint)

            input_data = loaded_graph.get_tensor_by_name('input:0')
            predObj = loaded_graph.get_tensor_by_name('predictions:0')
            text_length = loaded_graph.get_tensor_by_name('text_length:0')
            summary_length = loaded_graph.get_tensor_by_name('summary_length:0')
            keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')
    
            #Multiply by batch_size to match the model's input parameters
            predictionsValue = sess.run(predObj, {input_data: [text]*self.batch_size, 
                                      summary_length: [np.random.randint(5,8)], 
                                      text_length: [len(text)]*self.batch_size,
                                      keep_prob: 1.0})[0] 

        # Remove the padding from the tweet
        pad = self.vocab_to_int["<PAD>"] 
        returnPredict =  " ".join([self.int_to_vocab[i] for i in predictionsValue if i != pad])
        return returnPredict


    '''
    validationRatio and testRatio should be in decimal values less than 1.0

    '''
    def loadData(self,validationRatio,testRatio=0):
        infile = open(INT_TO_VOCAB_FLNAME,'rb')
        self.int_to_vocab  = pickle.load(infile)
        infile.close()

        infile = open(VOCAB_TO_INT_FLNAME,'rb')
        self.vocab_to_int  = pickle.load(infile)
        infile.close()

        myArray = np.load(open(CLEAN_TXT_FLNAME, 'rb'),allow_pickle=True)

        self.clean_texts = myArray.tolist()

        self.word_embedding_matrix = np.load(open(WORD_EMBEDDING_FLNAME, 'rb'),allow_pickle=True)

        rawSynopsis = np.load(open(SYNOPSIS_FLNAME, 'rb'),allow_pickle=True).tolist()    
        rawtagLine = np.load(open(TAGLINE_FLNAME, 'rb'),allow_pickle=True).tolist()    

        lengthOfData = len(rawSynopsis)
        debug("Total Data set Size")
        debug(lengthOfData)
        if(testRatio == 0):
            debug("Spliting data into Train and validation")

            sizeOftrain = lengthOfData * (1-validationRatio)
            trainsize = math.ceil(sizeOftrain)

            debug(sizeOftrain)
            debug(trainsize)
            for i in range(lengthOfData): 
                if(trainsize > i):
                    self.sorted_synopsis.append(rawSynopsis[i])
                    self.sorted_tagline.append(rawtagLine[i])
                else:
                    self.sorted_synopsis_validation.append(rawSynopsis[i])
                    self.sorted_tagline_validation.append(rawtagLine[i])
        else :
            debug("Spliting data into Train ,validation and test")
            sizeOftrain = lengthOfData * (1-validationRatio-testRatio)
            trainsize = math.ceil(sizeOftrain)
            sizeOfVal = lengthOfData * (validationRatio)
            valsize = math.ceil(sizeOfVal)
            debug(sizeOftrain)
            debug(trainsize)
            debug(sizeOfVal)
            debug(valsize)

            for i in range(lengthOfData): 
                if(trainsize > i):
                    self.sorted_synopsis.append(rawSynopsis[i])
                    self.sorted_tagline.append(rawtagLine[i])
                elif((trainsize+valsize) > i):
                    self.sorted_synopsis_validation.append(rawSynopsis[i])
                    self.sorted_tagline_validation.append(rawtagLine[i])
                else:
                    self.sorted_synopsis_test.append(rawSynopsis[i])
                    self.sorted_tagline_test.append(rawtagLine[i])         
        debug("Load Completed")
        
    def text_to_seq(self,text):
        '''Prepare the text for the model'''
        text = util.clean_text(text)
        return [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in text.split()]
    
    def runValidation(self):
        predictionResults = []
        predictionScores=[]
        for index,inputSentence in enumerate(self.sorted_synopsis_validation):
            prectionResult = self.predictFromProcessedSentence(inputSentence)
            predictionResults.append(prectionResult)
            #print(inputSentence)
            #print(prectionResult)
            tag = self.getGenerateTagline(self.sorted_tagline_validation[index])
            score = rnnhelper.getScore(prectionResult,tag)
            predictionScores.append(score)
        debug('Validation Completed')
        return predictionResults,self.sorted_tagline_validation ,predictionScores

    def runTestDataSet(self):
        predictionResults = []
        predictionScores=[]
        for index,inputSentence in enumerate(self.sorted_synopsis_test):
            prectionResult = self.predictFromProcessedSentence(inputSentence)
            predictionResults.append(prectionResult)
            #print(inputSentence)
            #print(prectionResult)
            tag = self.getGenerateTagline(self.sorted_tagline_validation[index])
            score = rnnhelper.getScore(prectionResult,tag)
            predictionScores.append(score)
        debug('Validation Completed')
        return predictionResults,self.sorted_tagline_validation ,predictionScores
        