# setup imports
from inverted_index_gcp import *
import pandas as pd
from contextlib import closing
import nltk
from nltk.corpus import stopwords
import json
import math
import numpy as np
import re
import gensim

# some constants
TUPLE_SIZE = 6
TF_MASK = 2 ** 16 - 1  # Masking the 16 low bits of an integer
RE_WORD = re.compile(r"""[\#\@\w](['\-]?\w){2,24}""", re.UNICODE)

nltk.download('stopwords')

# project_path = "/content/IR Project Files"
project_path = "/home/omer_idgar1/IR Project Files"

model = gensim.models.KeyedVectors.load_word2vec_format(f"{project_path}/Model/Word2VecWiki.bin", binary=True)

title_bins = f"{project_path}/Postings/postings_title_gcp/"
body_bins = f"{project_path}/Postings/postings_body_gcp/"
anchor_bins = f"{project_path}/Postings/postings_anchor_gcp/"

title_index = InvertedIndex.read_index(title_bins, 'index')
body_index = InvertedIndex.read_index(body_bins, 'index')
anchor_index = InvertedIndex.read_index(anchor_bins, 'index')

title_words = set(title_index.df.keys())
body_words = set(body_index.df.keys())
anchor_words = set(anchor_index.df.keys())

page_views = pd.read_pickle(f"{project_path}/Page Views/pageviews-202108-user.pkl")

with open(f"{project_path}/Page Rank/page_rank_data.json") as file:
    page_rank = json.load(file)

id_title = pd.read_pickle(f"{project_path}/ID Title/id_title_dict.pickle")

# Handle tokenize function
english_stopwords = frozenset(stopwords.words('english'))
corpus_stopwords = ["category", "references", "also", "external", "links",
                    "may", "first", "see", "history", "people", "one", "two",
                    "part", "thumb", "including", "second", "following",
                    "many", "however", "would", "became"]

all_stopwords = english_stopwords.union(corpus_stopwords)


def tokenize(text):
    """
    This function aims in tokenize a text into a list of tokens. Moreover, it filter stopwords.

    Parameters:
    -----------
    text: string , representing the text to tokenize.

    Returns:
    -----------
    list of tokens (e.g., list of tokens).
    """
    list_of_tokens = [token.group() for token in RE_WORD.finditer(text.lower()) if token.group() not in all_stopwords]
    return list_of_tokens


def read_posting_list(inverted, w, bins_path):
    """
    read posting list of specific word
    Args:
        inverted: the index in which we are searching the word
        w: the desired word
        bins_path: the path of the posting's bins

    Returns:
        list of tuples
            each tuple represent doc and tf in this doc
    """
    with closing(MultiFileReader()) as reader:
        locs = inverted.posting_locs[w]
        res = []
        for loc, offset in locs:
            res.append((bins_path + loc, offset))
        b = reader.read(res, inverted.df[w] * TUPLE_SIZE)
        posting_list = []
        for i in range(inverted.df[w]):
            doc_id = int.from_bytes(b[i * TUPLE_SIZE:i * TUPLE_SIZE + 4], 'big')
            tf = int.from_bytes(b[i * TUPLE_SIZE + 4:(i + 1) * TUPLE_SIZE], 'big')
            posting_list.append((doc_id, tf))
        return posting_list


def id2title(ids):
    """
    function that gives for each id its title
    Args:
        ids: list of doc_id

    Returns:
        list of tuples
            each tuple represent doc_id its title
    """
    res = []
    for doc in ids:
        try:
            doc = doc[0]
        except:
            pass
        title = id_title.get(doc, None)
        res.append((doc, title))
    return res


def search_in_body(query):
    """
    search the query and retrieve the desired documents using tf-idf and cosine similarity
    Args:
        query: list of tokens representing the query

    Returns:
        list of doc_id sorted in descending order of cosine similarity with the query
    """
    query_counter = Counter(query)  # count the num of occurrences of each token in the query

    epsilon = .0000001

    # dict of ints, represent the tfidf value of each token in the query , {"hello":0.52, "john":0.32}
    query_tfidf = {}

    # dict of lists, represent the tfidf value of each token in each document, {doc1:[("hello", 0.67),("john", 0.53)]}
    document_tfidf = {}

    # loop over each token and calculate its tf-idf value in the query,
    # than also calculate the tf-idf of the word in the documents its exists
    for token in query_counter.keys():
        if token in body_index.df.keys():  # avoid terms that do not appear in the index.
            tf = query_counter[token] / len(query)  # term frequency divided by the length of the query
            df = body_index.df[token]
            idf = math.log((len(body_index.DL)) / (df + epsilon), 10)  # smoothing

            token_tfidf = tf * idf
            query_tfidf[token] = token_tfidf  # update the tf-idf of the token in the query

            posting = read_posting_list(body_index, token, body_bins)
            for doc_id, frequency in posting:
                doc_tfidf = idf * frequency / body_index.DL[doc_id]
                document_tfidf[doc_id] = document_tfidf.get(doc_id, [])
                document_tfidf[doc_id].append((token, doc_tfidf))  # update the tf-idf of the token each document

    # dict represent the cosine similarity of the query with each document
    doc_values = {}

    # loop over each doc_id and its token-tfidf
    for doc_id, tokens_tfidf in document_tfidf.items():
        numerator = 0
        doc_values[doc_id] = 0

        # calculate the numerator by dot product of the doc values and query values
        for token, tfidf in tokens_tfidf:
            numerator += tfidf * query_tfidf[token]
        denominator = body_index.DL[doc_id] * len(query)
        if denominator != 0:
            doc_values[doc_id] = numerator / denominator

    # return the documents ordered by cosine similarity
    top = [doc_id for doc_id, value in sorted(doc_values.items(), key=lambda item: item[1], reverse=True)]
    return top


def search_in(place_to_search, query):
    """
    Generic functions that returns for the query, the number of query words
    that appear in the title ordered in  descending order
    Args:
        place_to_search: str(title/body/anchor), where we want to search the tokens
        query: list of tokens representing the query

    Returns:
        list of tuples
            each tuple represent the doc_id and the num of query words appear in the title
    """
    if place_to_search == "body":
        index_words = body_words
        index = body_index
        bins = body_bins
    elif place_to_search == "title":
        index_words = title_words
        index = title_index
        bins = title_bins
    elif place_to_search == "anchor":
        index_words = anchor_words
        index = anchor_index
        bins = anchor_bins

    # count the num of occurrences of each token in the query
    words_counter = Counter()
    tokenized = np.unique(query)

    # loop for each token, read its posting list
    # and update the num of occurrences of the documents for the total tokens
    for word in tokenized:
        if word in index_words:
            posting = read_posting_list(index, word, bins)
            docs = Counter([doc_id for doc_id, tf in posting])
            words_counter += docs
    return words_counter.most_common()


def generate_query(query, n=10):
    """
    gets a query list, and populate it with similar words using our word2vec model
    Args:
        tokenized: list of tokens representing the query
        n: amount of similar words to add for each word in the query

    Returns:
        list represent the updated query
    """
    words_to_search = [token for token in query]  # list that holding the updated query
    for token in query:
        try:
            similar_words = model.most_similar([token],
                                               topn=n)  # get the top `n` similar words of specific token in query
            for word, cos_sim in similar_words:
                if cos_sim < 0.5:  # if cos sim is not lower enough
                    break
                words_to_search.append(word)
        except Exception:
            pass

    return words_to_search


def merge_title_body(title_query, body_query, n):
    """
    returns the merge result of the titles docs and the body docs,
    while populating the query for the body index
    Args:
        title_query: the query we send for the title index
        body_query: the query we send for the body index
        n: amount of similar words to add for each word in the body query

    Returns:
        list of tuples, each tuple is doc_id, title
    """
    res_in_title = id2title(search_in("title", title_query))  # doc results of search in title
    new_body_query = generate_query(body_query, n)  # updated query for the body
    res_in_body = id2title(search_in("body", new_body_query)[:100])  # doc results of search in body

    # merge the results of the two indices
    dict_res_title = {doc_id: title for doc_id, title in res_in_title}
    dict_res_body = {doc_id: title for doc_id, title in res_in_body}

    inter = dict_res_title.keys() & dict_res_body.keys()

    top = [(overlap_doc_id, dict_res_body[overlap_doc_id]) for overlap_doc_id in inter]

    if not top:
        len_res_title = len(res_in_title)
        top = res_in_title[:min(len_res_title, 100)] + res_in_body[:100 - min(len_res_title, 100)]

    return top
