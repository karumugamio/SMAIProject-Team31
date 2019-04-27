import csv
import logging
import string

import numpy as np
import pandas as pd

from model.RNNModel import RNNSeq2SeqModel

logging.basicConfig(level=logging.DEBUG,format='%(levelname)s %(message)s')

def debug(arg):
    logging.debug(arg)

def info(arg):
    logging.info(arg)

def warn(arg):
    logging.warn(arg)

def run():
    debug("CSE471 SMAI - Spring 2019 - Project - Team 31")
    model = RNNSeq2SeqModel()
    
    model.loadDataFromLocalFile()

    model.fit()
    input_sentence = "Story of lawyer who loses ability to lie to people because of wish made by his son on birthday evening. this leads to alot of disaster and finally he relize that"
    prediction = model.predict(input_sentence)

    debug('prediction ['+str(prediction)+']')
    
    debug("____________________THE END____________________")



if __name__ == "__main__":
	run()