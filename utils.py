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
    "satan's cheerleaders": "SC",``
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

    example = pattern.sub(_repl, example)

    ##### YOUR CODE ENDS HERE ######

    return example
