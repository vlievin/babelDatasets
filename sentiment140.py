import os
import math
from time import time
import wget
import re
import os 
import itertools
from random import shuffle
import copy
import gzip
import zipfile
import numpy as np
import pandas as pd
from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from babelDatasets.utils import padding_merge, pad, EncoderDecoder

_DIGIT_RE = re.compile("\d")
_SEPARATOR_RE = re.compile("[?!.]")
_USERNAME_RE = re.compile("(?<=^|(?<=[^a-zA-Z0-9-_\.]))@([A-Za-z]+[A-Za-z0-9]+)")
_URL_RE = re.compile("http\S+")
_VOCAB_PATH_ = "vocab.dat"
_BLOCK_SIZE = 5000
_TRAIN_BLOCKS_PATH = "training_blocks"
_TEST_BLOCKS_PATH = "testing_blocks"
_PAD_SYMBOL = '*'


_ZIP_NAME = "sentiment140.zip"
_URL = "http://cs.stanford.edu/people/alecmgo/trainingandtestdata.zip"
_TEST_CSV_NAME = "testdata.manual.2009.06.14.csv"
_TRAIN_CSV_NAME = "training.1600000.processed.noemoticon.csv" 

def download(url, data_dir):
    """
    download source data 
    Args:
        url (str): url
        data_dir (str): name of the directory where to save the files
    Returns:
        data_path (str): path to file
    """
    data_path = os.path.join(data_dir, _ZIP_NAME)
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_path):
        data_path = wget.download(url, data_path)
    return data_path

def extractFile2Dir(path_to_zip_file,directory_to_extract_to):
    zip_ref = zipfile.ZipFile(path_to_zip_file, 'r')
    zip_ref.extractall(directory_to_extract_to)
    zip_ref.close()

def clean_sentence(sentence):
    #sentence =  sentence.lower()
    #sentence = re.sub(_DIGIT_RE,"0",sentence) # digits to zeros
    sentence = re.sub(_USERNAME_RE,"@user", sentence)
    sentence = re.sub(_URL_RE,"@url", sentence)
    sentence = re.sub('[^a-zA-Z.(-@:;#,.\)\(?!=$^ )]+', '', sentence)
    #sentence = re.sub("[\(\[].*?[\)\]]|&#00|\x1f", " ", sentence) # remove text in parenthesis and special character
    #sentence = re.sub('^W+\,|[%&()*\"\:\;\[\]\-\#\\\_<>\|\`\^\/\=\{\}\~€$@\+]',' ', sentence )
    #sentence = re.sub("[ ]{2,}" , " ", sentence)
    sentence = entence =  sentence.strip()
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

def loadData(data_dir, train):
    csv_name = _TRAIN_CSV_NAME if train else _TEST_CSV_NAME
    if not os.path.exists( os.path.join(data_dir, csv_name) ):
        zip_path = download(_URL,data_dir)
        extractFile2Dir(zip_path,data_dir)
    df =  pd.read_csv(os.path.join(data_dir, csv_name), encoding='latin-1',header=None,names=["polarity","id","date","query","user","text"])
    df['text'] = df['text'].map(lambda x: clean_sentence(x))
    return df

def getVoc(data_dir):
    voc_path = os.path.join(data_dir, _VOCAB_PATH_)
    if not os.path.exists(voc_path):
        sentences = list(loadData(data_dir, train=False)['text'].values) + list(loadData(data_dir, train=True)['text'].values )
        vocab = getVocabularyFromSentences(sentences)
        with open(os.path.join(data_dir,_VOCAB_PATH_), 'w') as vocab_file:
            [vocab_file.write("%s\n" % item) for item in vocab]
    return open(voc_path).read().splitlines()

def writeBlocks(sentences, out_directory, block_size):
    """
    Write a list of sentences into blocks of equal length as text files names as 'block-<index>.txt'
    Args:
        sentences (iter): iterator of sentences
        out_directory (str): path to the directoy to write the data to
    """
    if not os.path.exists(out_directory):
        os.makedirs(out_directory)
    # write blocks
    index = 0
    while True:
        block = list(itertools.islice(sentences,block_size))
        if len(block) < block_size:
            break
        block_path = 'block-'+str(index)+'.txt'
        block_path = os.path.join(out_directory , block_path)
        with open(block_path, 'w') as file:
            [file.write("%s\n" % item) for item in block]
        index += 1
    
def getBlockPaths(directory):
    """
    get the list of paths from the target directory
    """
    return [ os.path.join(directory,u) for u in os.listdir(directory) if 'block-' == u[:6]]

def readLinesFromFile(path):
    """
    get generator from text file
    """
    f = open(path)
    for l in f:
        yield l.replace('\n',"")
        
def shuffleGenerator(L):
    """ 
    shiffle iterator (requires to consume it)
    """
    L = list(L)
    shuffle(L)
    return iter(L)
        
class Sentiment140(Dataset):
    """
    Twitter Sentiment140 Dataset
    """
    def __init__(self, data_directory, train=True, train_ratio = 0.95, max_sentence_size=128):
        # download and load data 
        blocks_directory = _TRAIN_BLOCKS_PATH #if train else _TEST_BLOCKS_PATH
        blocks_directory = os.path.join(data_directory,blocks_directory)
        if not os.path.exists(blocks_directory):
            sentences = loadData(data_directory,train)['text'].values
            sentences = [s for s in sentences if len(s) < max_sentence_size]
            writeBlocks(iter(sentences), blocks_directory, block_size = min(_BLOCK_SIZE,len(sentences))) 
        self.vocab = getVoc(data_directory)
        # load objects
        self.encoderDecoder = EncoderDecoder(self.vocab, _PAD_SYMBOL)
        self.blockPaths = list(getBlockPaths(blocks_directory))
        if train:
            self.blockPaths = self.blockPaths[:int(train_ratio*len(self.blockPaths))]
        else:
            self.blockPaths = self.blockPaths[int(train_ratio*len(self.blockPaths)):]
        self.current_blockPaths = shuffleGenerator(copy.copy(self.blockPaths))
        self.current_sentences = shuffleGenerator(readLinesFromFile(next(iter(self.current_blockPaths))))

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
        #if self.min_sentence_size is not None:
        #    while not (len(sentence) >= self.min_sentence_size and len(sentence) <= self.max_sentence_size):
        #        sentence = self.getNextSentence()
        return self.encoderDecoder.encode(sentence)