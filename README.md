# IR-Wikipedia-Search-Engine
In this project we build a search engine for the entire Wikipedia corpus. The engine workflow is provided bellow:

![Search Engine Diagram](https://user-images.githubusercontent.com/87119884/148761838-7d7537b8-09d6-4e53-a371-6d3a3459b2fd.png)

# Search body:
Returns up to a 100 search results for the query using TFIDF AND Csine Similarity of the body articles.
# Search title:
Returns all search results that contain a query word in the title of articles, ordered in descending order of the number of query words that appear in the title. For example, a document with a title that matches two of the query words will be ranked before a document with a title that matches only one query term.
# Searc anchor:
Returns all search results that contain a query word in the anchor text of articles, ordered in descending order of the number of query words that appear in the text linking to the page. For example, a document with a anchor text that matches two of the query words will be ranked before a document with anchor text that matches only one query term.
# Search:
In this part we create a Word2Vec model for the entire Wikipedia corpus. With this model, we can find semantics between the query words provided by the user and the most similar words in the corpus for the word in the query. Moreover, with the model we can found similarity between words in the query (for example, for the words “information” and “retrieval” we get a high similarity score.
We create the model using the genism package and saved the trained model in a bin file and upload it to the bucket. 
First, we tokenize the query and check what is the number of words the query contain. With this information we know in which way to use our model.
