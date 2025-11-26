# src/tokenizer.py
from konlpy.tag import Komoran

tagger = Komoran()

def extract_terms(text):
    tokens = tagger.pos(text)
    return [w.lower() if t == 'SL' else w for w, t in tokens
            if t in {'NNG', 'NNP', 'SL'}]