import datasets
from datasets import load_dataset
from transformers import AutoTokenizer
from torch.utils.data import DataLoader
from transformers import AutoModelForSequenceClassification
from torch.optim import AdamW
from transformers import get_scheduler
import torch
from tqdm.auto import tqdm
import evaluate
import random
import argparse
from nltk.corpus import wordnet
from nltk import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
import re

random.seed(0)


def example_transform(example):
    example["text"] = example["text"].lower()
    return example


### Rough guidelines --- typos
# For typos, you can try to simulate nearest keys on the QWERTY keyboard for some of the letter (e.g. vowels)
# You can randomly select each word with some fixed probability, and replace random letters in that word with one of the
# nearest keys on the keyboard. You can vary the random probablity or which letters to use to achieve the desired accuracy.


### Rough guidelines --- synonym replacement
# For synonyms, use can rely on wordnet (already imported here). Wordnet (https://www.nltk.org/howto/wordnet.html) includes
# something called synsets (which stands for synonymous words) and for each of them, lemmas() should give you a possible synonym word.
# You can randomly select each word with some fixed probability to replace by a synonym.


def custom_transform(example):
    ################################
    ##### YOUR CODE BEGINGS HERE ###

    # Design and implement the transformation as mentioned in pdf
    # You are free to implement any transformation but the comments at the top roughly describe
    # how you could implement two of them --- synonym replacement and typos.

    # You should update example["text"] using your transformation


    # 1. Phrase -> acronym mapping.
    PHRASE_TO_ACRONYM = {
    # Places and orgs
    "united states": "US",
    "new york city": "NYC",
    "new mexico": "NM",
    "public broadcasting service": "PBS",
    "hallmark channel": "HMC",
    "hallmark movie channel": "HMC",
    "lifetime movie network": "LMN",
    "brain damage films": "BDF",

    # Education
    "general certificate of secondary education": "GCSE",

    # Formats / tech
    "video home system": "VHS",
    "digital versatile disc": "DVD",
    "three dimensional": "3D",
    "two dimensional": "2D",
    "sport utility vehicle": "SUV",

    # Film craft / genre
    "special effects": "SFX",
    "visual effects": "VFX",
    "computer generated imagery": "CGI",
    "original soundtrack": "OST",
    "background music": "BGM",
    "romantic comedy": "ROM-COM",
    "science fiction": "sci-fi",
    "point of view": "POV",

    # Movie titles 
    "the zombie chronicles": "TZC",
    "camp blood": "CB",
    "love's abiding joy": "LAJ",
    "love come softly": "LCS",
    "the return": "TR",
    "star 80": "S80",
    "death of a centerfold: the dorothy stratten story": "DOC",
    "the last picture show": "TLPS",
    "paper moon": "PM",
    "what's up, doc": "WUD",
    "breakfast at tiffany's": "BAT",
    "my fair lady": "MFL",
    "love among thieves": "LAT",
    "mystery science theater 3000": "MST3K",
    "close encounters of the third kind": "CE3K",
    "final justice": "FJ",
    "satan's cheerleaders": "SC",
    }
    # Ensure keys are lowercase (defensive)
    _lower_map = {k.lower(): v for k, v in PHRASE_TO_ACRONYM.items()}

    # Sort phrases by length so longer phrases match first
    phrases_sorted = sorted(_lower_map.keys(), key=len, reverse=True)
    # Build regex pattern that matches any phrase as a whole "word chunk"
    # re.escape handles apostrophes, colons, etc.
    pattern = re.compile(
        r"\b(" + "|".join(map(re.escape, phrases_sorted)) + r")\b",
        re.IGNORECASE,
    )

    def _repl(match: re.Match) -> str:
        phrase = match.group(0)
        return _lower_map[phrase.lower()]

    # Apply transformation to the text field
    example["text"] = pattern.sub(_repl, example["text"])

    # 2. Typo transformation - simulate keyboard typos
    # Define QWERTY keyboard layout and nearest keys
    QWERTY_NEIGHBORS = {
        'a': ['q', 'w', 's', 'z'],
        'b': ['v', 'g', 'h', 'n'],
        'c': ['x', 'd', 'f', 'v'],
        'd': ['s', 'e', 'r', 'f', 'c', 'x'],
        'e': ['w', 'r', 'd', 's'],
        'f': ['d', 'r', 't', 'g', 'v', 'c'],
        'g': ['f', 't', 'y', 'h', 'b', 'v'],
        'h': ['g', 'y', 'u', 'j', 'n', 'b'],
        'i': ['u', 'o', 'k', 'j'],
        'j': ['h', 'u', 'i', 'k', 'm', 'n'],
        'k': ['j', 'i', 'o', 'l', 'm'],
        'l': ['k', 'o', 'p'],
        'm': ['n', 'j', 'k', 'l'],
        'n': ['b', 'h', 'j', 'm'],
        'o': ['i', 'p', 'l', 'k'],
        'p': ['o', 'l'],
        'q': ['w', 'a'],
        'r': ['e', 't', 'f', 'd'],
        's': ['a', 'w', 'e', 'd', 'x', 'z'],
        't': ['r', 'y', 'g', 'f'],
        'u': ['y', 'i', 'j', 'h'],
        'v': ['c', 'f', 'g', 'b'],
        'w': ['q', 'e', 's', 'a'],
        'x': ['z', 's', 'd', 'c'],
        'y': ['t', 'u', 'h', 'g'],
        'z': ['a', 's', 'x'],
    }
    
    # Probability of introducing a typo in a word
    TYPO_PROBABILITY = 0.15  # 15% chance per word
    # Probability of replacing a letter within a selected word
    LETTER_REPLACE_PROB = 0.3  # 30% chance per letter in selected word
    
    def introduce_typos(text):
        # Tokenize into words while preserving structure
        words = word_tokenize(text)
        transformed_words = []
        
        for word in words:
            # Skip if not a word (punctuation, etc.)
            if not word.isalpha():
                transformed_words.append(word)
                continue
            
            # Decide if we should introduce typos in this word
            if random.random() < TYPO_PROBABILITY:
                word_chars = list(word)
                has_typo = False
                
                # Try to replace letters with nearby keyboard keys
                for i, char in enumerate(word_chars):
                    char_lower = char.lower()
                    # Only process alphabetic characters
                    if char_lower.isalpha() and char_lower in QWERTY_NEIGHBORS:
                        if random.random() < LETTER_REPLACE_PROB:
                            # Get nearby keys
                            neighbors = QWERTY_NEIGHBORS[char_lower]
                            if neighbors:
                                # Pick a random neighbor
                                replacement = random.choice(neighbors)
                                # Preserve case
                                if char.isupper():
                                    replacement = replacement.upper()
                                word_chars[i] = replacement
                                has_typo = True
                
                # If we made a typo, reconstruct the word
                if has_typo:
                    transformed_words.append(''.join(word_chars))
                else:
                    transformed_words.append(word)
            else:
                transformed_words.append(word)
        
        # Reconstruct text using detokenizer
        detokenizer = TreebankWordDetokenizer()
        return detokenizer.detokenize(transformed_words)
    
    # Apply typo transformation
    example["text"] = introduce_typos(example["text"])

    # 3. Filler / hedging phrases transformation
    # Define filler and hedging phrases
    FILLER_PHRASES = [
        "to be honest",
        "in my opinion",
        "if I am being honest",
        "if I'm being honest",
        "kind of",
        "really",
        "actually",
        "I think",
        "I believe",
        "I feel",
        "I guess",
        "I suppose",
        "to be fair",
        "honestly",
        "frankly",
        "personally",
        "in all honesty",
        "truth be told",
        "I have to say",
        "I must say",
        "sort of",
        "pretty much",
        "more or less",
    ]
    
    # Probability of inserting a filler phrase
    FILLER_PROBABILITY = 0.25  # 25% chance per sentence
    # Probability of inserting before adjectives/adverbs
    BEFORE_WORD_PROB = 0.15  # 15% chance before certain words
    
    def add_filler_phrases(text):
        # Split into sentences (simple approach using periods, exclamation, question marks)
        
        # Split by sentence boundaries but keep the punctuation
        sentences = re.split(r'([.!?]+)', text)
        # Recombine sentences with their punctuation
        sentence_pairs = []
        for i in range(0, len(sentences) - 1, 2):
            if i + 1 < len(sentences):
                sentence_pairs.append((sentences[i], sentences[i + 1]))
            else:
                sentence_pairs.append((sentences[i], ""))
        
        transformed_sentences = []
        
        for sentence, punctuation in sentence_pairs:
            if not sentence.strip():
                transformed_sentences.append(sentence + punctuation)
                continue
            
            # Decide if we should add a filler phrase at the start of the sentence
            if random.random() < FILLER_PROBABILITY:
                filler = random.choice(FILLER_PHRASES)
                # Capitalize first letter if sentence starts with capital
                if sentence and sentence[0].isupper():
                    filler = filler.capitalize()
                sentence = filler + ", " + sentence
            
            # Also add filler words before certain words (adjectives, strong verbs)
            # Common patterns: before "very", "really", "extremely", etc.
            intensity_words = ["very", "really", "extremely", "incredibly", "absolutely", 
                              "completely", "totally", "quite", "rather", "pretty"]
            
            words = sentence.split()
            new_words = []
            for i, word in enumerate(words):
                # Clean word for comparison (remove punctuation)
                word_clean = re.sub(r'[^\w]', '', word.lower())
                
                # Sometimes add filler before intensity words
                if word_clean in intensity_words and random.random() < BEFORE_WORD_PROB:
                    filler = random.choice(["kind of", "sort of", "really", "pretty much"])
                    new_words.append(filler)
                
                new_words.append(word)
            
            sentence = " ".join(new_words)
            transformed_sentences.append(sentence + punctuation)
        
        return " ".join(transformed_sentences)
    
    # Apply filler phrases transformation
    example["text"] = add_filler_phrases(example["text"])

    ##### YOUR CODE ENDS HERE ######

    return example
