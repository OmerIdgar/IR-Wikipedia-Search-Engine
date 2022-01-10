import os
from fastparquet import ParquetFile
import snappy
import re
from nltk.corpus import stopwords
import time
from gensim.models.doc2vec import Word2Vec, TaggedDocument

english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

all_stopwords = english_stopwords.union(corpus_stopwords)


def snappy_decompress(data, uncompressed_size):
    return snappy.decompress(data)


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , represting the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


class MySentences(object):
    """
    Helper Class for iterating the entire corpus without loading
    the whole corpus to the main memory.
    """
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):

        for fname in os.listdir(self.dirname):
            x = time.time()

            pf = ParquetFile(os.path.join(self.dirname, fname))
            dff = pf.to_pandas()
            texts = dff['text']
            for text in texts:
                tokenized = tokenize(text)
                if len(tokenized) >= 50:
                    yield tokenized




start = time.time()

start = time.time()
sentences = MySentences('wiki/')  # a memory-friendly iterable
model = Word2Vec(vector_size=300, window=8, epochs=10, min_count=10, workers=10)
model.build_vocab(sentences, progress_per=10000)
print(f'total time to build the vocabulary: {(time.time() - start) / 60} minutes!')

start = time.time()

model.train(sentences, epochs=model.epochs, total_examples=model.corpus_count)
print(f'total time to train the model: {(time.time() - start) / 60} minutes!')
model.wv.save_word2vec_format("Word2VecWiki", binary=True)
