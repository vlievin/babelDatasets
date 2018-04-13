"""
Amazon Reviews dataset

http://jmcauley.ucsd.edu/data/amazon/

"""

import torch
from torch.utils.data import Dataset, DataLoader
import sqlite3
import dataset
import wget
import re
import os 
import gzip
import itertools
import pandas as pd
import numpy as np
import copy
import uuid
from tqdm import tqdm
from random import shuffle
from babelDatasets.utils import padding_merge, pad, EncoderDecoder, chunks

_DATA_ROOT = "/scratch/babel/"
_DATASET_URL = "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Movies_and_TV_5.json.gz"
                #("reviews_Books_5" ,"http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Books_5.json.gz")
                #("pet-supplies.json.gz", "http://snap.stanford.edu/data/amazon/productGraph/categoryFiles/reviews_Pet_Supplies_5.json.gz")]
    
_DIGIT_RE = re.compile("\d")
_SEPARATOR_RE = re.compile("[?!.]")
_MINIMUM_SENTENCE_LENGTH = 30
_MAXIMUM_SENTENCE_LENGTH = 300
_MAXIMUM_NUMBER_OF_SENTENCES = 10000000
_VOCAB_PATH_ = "vocab.dat"
_BLOCK_SIZE = 1000
_PAD_SYMBOL = '*'


class AmazonReviews(Dataset):
    """
    AmazonReviews dataset.
    NB: This dataset is not safe for use with multiple threads (PyTorch Dataloader)
    """
    def __init__(self, url = _DATASET_URL, data_directory = _DATA_ROOT, min_sentence_size=None,max_sentence_size=None, train_ratio=0.8, train=True):
        # sentence minimum
        self.min_sentence_size = min_sentence_size
        self.max_sentence_size = max_sentence_size
        if self.min_sentence_size is not None or self.max_sentence_size is not None:
            self.min_sentence_size = 0 if self.min_sentence_size is None else self.min_sentence_size
            self.max_sentence_size = 999999999 if self.max_sentence_size is None else self.max_sentence_size
        # build data 
        blocks_directory = os.path.join(data_directory,url.split('/')[-1].split('.')[0])
        if not os.path.exists(blocks_directory):
            writeBlocks(url, blocks_directory) 
        self.vocab = readVocab(blocks_directory)
        # load objects
        self.encoderDecoder = EncoderDecoder(self.vocab, _PAD_SYMBOL)
        self.blockPaths = list(getBlockPaths(blocks_directory))
        if train:
            self.blockPaths = self.blockPaths[:int(train_ratio*len(self.blockPaths))]
        else:
            self.blockPaths = self.blockPaths[int(train_ratio*len(self.blockPaths)):]
        self.current_blockPaths = shuffleGenerator(copy.copy(self.blockPaths))
        self.current_sentences = shuffleGenerator(readLinesFromFile(next(iter(self.current_blockPaths))))

    def __len__(self):
        return _BLOCK_SIZE * len(self.blockPaths)

    def getNextSentence(self):
        sentence = next(iter(self.current_sentences), None)
        if sentence is None:
            next_block = next(iter(self.current_blockPaths), None)
            if next_block is None:
                self.current_blockPaths = shuffleGenerator(copy.copy(self.blockPaths))
                next_block = next(iter(self.current_blockPaths), None)
            self.current_sentences = shuffleGenerator(readLinesFromFile(next(iter(self.current_blockPaths))))
            sentence = next(iter(self.current_sentences), None)
        return sentence
            
    def __getitem__(self, idx):
        sentence = self.getNextSentence()
        if self.min_sentence_size is not None:
            while not (len(sentence) >= self.min_sentence_size and len(sentence) <= self.max_sentence_size):
                sentence = self.getNextSentence()
        return self.encoderDecoder.encode(sentence)


def shuffleGenerator(L):
    """ 
    shiffle iterator (requires to consume it)
    """
    L = list(L)
    shuffle(L)
    return iter(L)
    
def readLinesFromFile(path):
    """
    get generator from text file
    """
    f = open(path)
    for l in f:
        yield l.replace('\n',"")
        
def getBlockPaths(directory):
    """
    get the list of paths from the target directory
    """
    return [ os.path.join(directory,u) for u in os.listdir(directory) if 'block-' == u[:6]]

def readVocab(directory):
    """
    read vocabulary from file
    """
    path = os.path.join(directory,"vocab.dat")
    return open(path).read().splitlines()

def getCleanedSentencesFromUrl(url):
    """
    get cleaned sentences from file
    """
    f = url.split("/")[-1]
    path = download(url,f)
    data = parse(path)
    reviews = (d['reviewText'] for d in data)
    all_sentences = reviews2sentences(reviews,min_sentence_size=_MINIMUM_SENTENCE_LENGTH,max_sentence_size=_MAXIMUM_SENTENCE_LENGTH)
    return (clean_sentence(sent) for sent in all_sentences)

def writeBlocks(url, out_directory):
    """
    Download an Amazon Reviews item, read, clean and store into blocks of equal length as text files names as 'block-<index>.txt'
    Args:
        url (str): Amazon Reviews url to download from
        out_directory (str): path to the directoy to write the data to
    """
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # make vocabulary and write to file
    vocab = getVocabularyFromSentences(getCleanedSentencesFromUrl(url))
    with open(os.path.join(out_directory,_VOCAB_PATH_), 'w') as vocab_file:
        [vocab_file.write("%s\n" % item) for item in vocab]
    # write blocks
    index = 0
    sentences = getCleanedSentencesFromUrl(url)
    while True:
        block = list(itertools.islice(sentences,_BLOCK_SIZE))
        if len(block) < _BLOCK_SIZE:
            break
        block_path = 'block-'+str(index)+'.txt'
        block_path = os.path.join(out_directory , block_path)
        with open(block_path, 'w') as file:
            [file.write("%s\n" % item) for item in block]
        index += 1
        
        
def download(url, filename):
    """
    download source data 
    Args:
        url (str): url
        filename (str): name to save the file to
    Returns:
        data_path (str): path to file
    """
    data_path = os.path.join(_DATA_ROOT, filename)
    if not os.path.exists(_DATA_ROOT):
        os.makedirs(_DATA_ROOT)
    if not os.path.exists(data_path):
        data_path = wget.download(url, data_path)
    return data_path

def parse(path):
    """
    parse gz file
    Args:
        path (str): path to file
    Return:
        generator
    """
    g = gzip.open(path, 'rb')
    for l in g:
        yield eval(l)

def getDF(path):
    """
    get Pandas dataframe from file
    Args:
        path (str): file
    Returns:
        data (pd.DataFrame)
    """
    i = 0
    df = {}
    for d in parse(path):
        df[i] = d
        i += 1
    return pd.DataFrame.from_dict(df, orient='index')

def reviews2sentences(reviews, min_sentence_size=0, max_sentence_size=999999):
    """
    convert the list of reviews into a flattened list of sentences
    Args:
        reviews (list(str)): list of reviews
    Returns:
        sentences (list(str)): list of sentences
        min_sentence_size (int) : minimum sentence size
        max_sentence_size (int) : maximum sentence size
        
    """
    return (sent+'.' for review_sentences in reviews for sent in re.split(_SEPARATOR_RE, review_sentences) if len(sent) >= min_sentence_size and len(sent) <= max_sentence_size)

def clean_sentence(sentence):
    """
    clean a sentence by removing some characters, cleaning redondant and leadning spaces
    Args:
        sentence (str): input sentence
    Returns:
        sentence (str): cleaned sentence
    """
    sentence =  sentence.strip()
    sentence =  sentence.lower()
    sentence = re.sub(_DIGIT_RE,"0",sentence) # digits to zeros
    sentence = re.sub("[\(\[].*?[\)\]]|&#00|\x1f", " ", sentence) # remove text in parenthesis and special character
    sentence = re.sub('^W+\,|[%&()*\"\:\;\[\]\-\#\\\_<>\|\`\^\/\=\{\}\@\~€$@\+]',' ', sentence )
    sentence = re.sub("[ ]{2,}" , " ", sentence)
    return sentence

def getVocabularyFromSentences(sentences,log=False):
    """
    Generate vocabulary from a list of sentences
    Args:
        sentences (list(str)): list of sentences
    Returns
        vocabulary (ordered_set(str)): ordred vocabulary of characters
    """
    return ['*'] + sorted(set( el for sentence in sentences for el in set(sentence)))
    

    
