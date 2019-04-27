import re
import warnings
import logging
import os

import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
import time
import pickle
import json

import Preprocessing.datautil as util
import model.RNNHelper as rnnhelper
import tensorflow as tf

CHECKPOINT = ''
CHECKPOINT_METAFILE = ''


logging.basicConfig(level=logging.WARN,format='%(asctime)s %(levelname)s %(message)s')

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
        self.sorted_summaries = []
        self.sorted_texts = []
        self.word_embedding_matrix=[]
        self.int_to_vocab = {}
        self.clean_texts = {}


        # Parameters related to Train the Model
        self.learning_rate_decay = 0.95
        self.min_learning_rate = 0.0005
        self.display_step = 20 # Check training loss after every 20 batches
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

        start = 2000
        end = start + 5000
        sorted_summaries_short = self.sorted_summaries[start:end]
        sorted_texts_short = self.sorted_texts[start:end]
        print("The shortest text length:", len(sorted_texts_short[0]))
        print("The longest text length:",len(sorted_texts_short[-1]))

        self.update_check = (len(sorted_texts_short)//self.batch_size//self.per_epoch)-1

        summary_update_loss = []
        checkpoint = "best_model.ckpt" 
        
        with tf.Session(graph=train_graph) as sess:
            sess.run(tf.global_variables_initializer())

            # If we want to continue training a previous session
            #loader = tf.train.import_meta_graph("./" + checkpoint + '.meta')
            #loader.restore(sess, checkpoint)
            for epoch_i in range(1, self.epochs+1):
                update_loss = 0
                batch_loss = 0 
                intValueForPad = self.vocab_to_int['<PAD>']    
                for batch_i, (summaries_batch, texts_batch, summaries_lengths, texts_lengths) in enumerate(rnnhelper.get_batches(sorted_summaries_short, sorted_texts_short, self.batch_size,intValueForPad)):
                    
                    start_time = time.time()
                    _, loss = sess.run([train_op, cost],{input_data: texts_batch,targets: summaries_batch,lr: self.learning_rate,summary_length: summaries_lengths,text_length: texts_lengths,keep_prob: self.keep_probability})          
                    batch_loss += loss
                    update_loss += loss
                    end_time = time.time()
                    batch_time = end_time - start_time
                    if batch_i % self.display_step == 0 and batch_i > 0:
                        debug('Epoch {:>3}/{} Batch {:>4}/{} - Loss: {:>6.3f}, Seconds: {:>4.2f}'.format(epoch_i,self.epochs,batch_i,len(sorted_texts_short) // self.batch_size, batch_loss / self.display_step,batch_time*self.display_step))
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

        checkpoint = "data/best_model.ckpt"

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

    def predictandPresent(self):
        debug("predict values")

    
    def run (self):
        
        # Create your own review or use one from the dataset
        #input_sentence = "I have never eaten an apple before, but this red one was nice. \
                        #I think that I will try a green apple next time."
        #text = text_to_seq(input_sentence)
        debug("runrun")

    def loadDataFromLocalFile(self):
        infile = open('data\int_to_vocab.pkl','rb')
        self.int_to_vocab  = pickle.load(infile)
        infile.close()

        infile = open("data\\vocab_to_int.pkl",'rb')
        self.vocab_to_int  = pickle.load(infile)
        infile.close()

        myArray = np.load(open('data\cleantxt.npy', 'rb'),allow_pickle=True)

        self.clean_texts = myArray.tolist()

        self.word_embedding_matrix = np.load(open('data\word_embedding_matrix.npy', 'rb'),allow_pickle=True)

        self.sorted_summaries = np.load(open('data\sorted_summaries.npy', 'rb'),allow_pickle=True).tolist()    
        self.sorted_texts = np.load(open('data\sorted_texts.npy', 'rb'),allow_pickle=True).tolist()    

        debug("Load Completed")
        
    def text_to_seq(self,text):
        '''Prepare the text for the model'''
        text = util.clean_text(text)
        return [self.vocab_to_int.get(word, self.vocab_to_int['<UNK>']) for word in text.split()]