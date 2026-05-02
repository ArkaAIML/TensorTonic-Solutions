import numpy as np
from typing import List, Dict

class SimpleTokenizer:
    """
    A word-level tokenizer with special tokens.
    """
    
    def __init__(self):
        self.word_to_id: Dict[str, int] = {}
        self.id_to_word: Dict[int, str] = {}
        self.vocab_size = 0
        
        # Special tokens
        self.pad_token = "<PAD>"
        self.unk_token = "<UNK>"
        self.bos_token = "<BOS>"
        self.eos_token = "<EOS>"
    
    def build_vocab(self, texts: List[str]) -> None:
        """
        Build vocabulary from a list of texts.
        Add special tokens first, then unique words.
        """
        self.word_to_id = {self.pad_token: 0, self.unk_token: 1, self.bos_token: 2, self.eos_token: 3}
        self.id_to_word = {0: self.pad_token, 1: self.unk_token, 2: self.bos_token, 3: self.eos_token}
        i = 4
        t = []
        
        # Lowercase and collect all words
        for x in texts:
            words = x.split()
            for w in words:
                w = w.lower()
                t.append(w) # Fixed parentheses
                
        t.sort() # Fixed sort

        # Assign IDs to alphabetical words
        for ww in t:
            if ww not in self.word_to_id:
                self.word_to_id[ww] = i # Fixed typo (ww instead of w)
                self.id_to_word[i] = ww
                i += 1
                    
        self.vocab_size = len(self.word_to_id)
    
    def encode(self, text: str) -> List[int]:
        """
        Convert text to list of token IDs.
        Use UNK for unknown words.
        """
        l = []
        words = text.lower().split() # Lowercase the string BEFORE splitting
        
        for w in words:
            if w in self.word_to_id:
                l.append(self.word_to_id[w])
            else:
                l.append(self.word_to_id[self.unk_token])
            
        return l
    
    def decode(self, ids: List[int]) -> str:
        """
        Convert list of token IDs back to text.
        """
        s = "" # Start empty
        
        for i in ids: # Loop over the actual variable
            if i in self.id_to_word:
                s += self.id_to_word[i] + " " # Use += to add to the string
            else :
                s+=self.unk_token+" "
        
        return s.strip() # .strip() removes the trailing extra space at the very end!