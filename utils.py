"""
utility modules for the babelDatasets module


"""

import numpy as np
import re

def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]
        
def pad(l, n):
    """
    pad sequence l with length n
    """
    return l[:n] + [0]*(n-len(l))
    
    
def padding_merge(batch_xs, fixed_lenght = None, receive_tuple=False):
    """
    merge a batch of sentences into tensor
    
    """
    batch_xs = list(zip(*batch_xs)) if receive_tuple else batch_xs
    sentences = batch_xs[0] if receive_tuple else batch_xs
    batch_lengths = [len(x) for x in sentences]
    if fixed_lenght is None:
        max_length = max(batch_lengths)
    else:
        max_length = fixed_lenght
    padded_sentences = [ pad(d, max_length) for d in sentences ]
    batch_weights = [ [ 1 if dd>0 else 0 for dd in d] for d in padded_sentences]
    if not receive_tuple:
        return np.asarray(padded_sentences), np.asarray(batch_weights)
    else:
        return np.asarray(padded_sentences), np.asarray(batch_weights), np.asarray(batch_xs[1])

def PaddingMerge(fixed_lenght = None, receive_tuple=False):
    return lambda x : padding_merge(x, fixed_lenght=fixed_lenght, receive_tuple=receive_tuple)


class EncoderDecoder():
    """
    A class to encode text to a sequence of ids or to decode a sequence of ids to text
    """
    def __init__(self, vocabulary, pad_symbol=''):
        """
        Load vocabulary
        Args:
            vocabulary (list):
            pad_symbol (char): padding symbol in the decoded sentence
        """
        self.char2encoding = {c:k for k,c in enumerate(vocabulary)}
        self.encoding2char = {k:c for k,c in enumerate(vocabulary)}
        self.pad_symbol = pad_symbol

    def encode(self, sentence):
        """
        Encode a sentence to a sequence of ids
        """
        return [self.char2encoding[c] for c in sentence if c in self.char2encoding.keys()]
        
    def decode(self, seq):
        """
        Decode a sequence of ids to a sentence
        """
        return "".join([ self.encoding2char[int(el)] for el in seq if el in self.encoding2char.keys()])
    
    def prettyDecode(self, seq):
        """
        Decode a sequence of ids to a sentence
        """
        sentence = self.decode(seq)
        return sentence.replace(self.pad_symbol,"")
    
        