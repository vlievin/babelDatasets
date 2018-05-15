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
_VOCAB_PATH = "vocab.dat"
_TRAIN_SENTENCES_PATH = "train_sentences.txt"
_TEST_SENTENCES_PATH = "test_sentences.txt"
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
    sentence = re.sub('[^a-zA-Z.(-@:;#,.\)\(?!=$^\' )]+', '', sentence)
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

def writeVocab(vocab,vocab_path):
    with open(vocab_path, 'w') as vocab_file:
        [vocab_file.write("%s\n" % item) for item in vocab]
    
def readVocab(vocab_path):
    return open(vocab_path).read().splitlines()

def writeEncodedSentences(sentences,filename):
    with open(filename, 'w') as f:
        for s in sentences:
            for _string in s:
                f.write(str(_string)+' ')
            f.write('\n')
def readEncodedSentences(filename):
    f = open(filename)
    for l in f:
        yield [ eval(x) for x in l.replace('\n',"").split(' ') if len(x)]
        
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
        train_sentences_path = os.path.join(data_directory,_TRAIN_SENTENCES_PATH)
        test_sentences_path = os.path.join(data_directory,_TEST_SENTENCES_PATH)
        vocab_path = os.path.join(data_directory,_VOCAB_PATH)
        if not os.path.exists(train_sentences_path) or not os.path.exists(test_sentences_path) or not os.path.exists(vocab_path):
            sentences = loadData(data_directory,train)['text'].values
            sentences = [s for s in sentences if len(s) < max_sentence_size]
            vocab = getVocabularyFromSentences(sentences)
            writeVocab(vocab,vocab_path)
            encoderDecoder = EncoderDecoder(vocab, _PAD_SYMBOL)
            sentences = [encoderDecoder.encode(s) for s in sentences]
            print('# sentences:', len(sentences))
            train_sentences = sentences[:int(train_ratio*len(sentences))]
            test_sentences = sentences[int(train_ratio*len(sentences)):]
            print('# train sentences:',len(train_sentences))
            print('# test_sentences sentences:',len(test_sentences))
            writeEncodedSentences(train_sentences, train_sentences_path)
            writeEncodedSentences(test_sentences, test_sentences_path)
        # load objects
        self.encoderDecoder = EncoderDecoder(readVocab(vocab_path), _PAD_SYMBOL)
        sentences_path = train_sentences_path if train else test_sentences_path
        self.sentences = list(readEncodedSentences(sentences_path))
        print("# N sents:", len(self.sentences), " train:", train , " sentences_path:", sentences_path)
    
    def __len__(self):
        return len(self.sentences)
            
    def __getitem__(self, idx):
        return self.sentences[idx]