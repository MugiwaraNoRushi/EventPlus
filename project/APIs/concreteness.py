import pandas as pd
import pickle
import sys
import os
import argparse
import json
import re
import nltk
from class_event_loader import EventAPIs
from nltk import tokenize
import numpy as np
from frameAxis import FrameAxis
from gensim.models import KeyedVectors
import gensim
import gensim.downloader
from test_on_raw_text import NumpyEncoder

# functions


def _sanitize(text: str) -> str:
    """Cleans text by removing whitespace, newlines and tabs
    """
    sanitized_text = " ".join(str(text).strip().split())
    return sanitized_text

def deEmojify(text):
    regrex_pattern = re.compile(pattern = "["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags = re.UNICODE)
    return regrex_pattern.sub(r'',text)

def preprocess_text(tweets):
    '''remove hashtags and urls'''
    tweets = tweets.str.replace(r'(?:\@|https?\://)\S+', ' ', regex=True)
    print(type(tweets))
    tweets = tweets.str.replace('\xa0', '')
    tweets = tweets.str.replace(r'[^\w\s]', ' ', regex=True)
    tweets = tweets.str.replace(r'[^a-zA-z\s]', ' ', regex=True)  # r'[^a-zA-z0-9\s]'
    # todo maybe also stem?
    '''remove stopwords'''
#     for s_word in stop_words:
#         tweets = tweets.str.replace(' ' + s_word + ' ', ' ')
    tweets = tweets.apply(_sanitize)
    return tweets

def remove_stopwords(tweet):
    return ' '.join([w for w in tweet.split() if w not in stopwords])

def MF_concreteness(combined_result, fa_con):
    if len(combined_result["tokens"]) >=5:
        sentence_df = pd.DataFrame([combined_result['sentence']], columns=['text'])
        sentence_df['prep_text'] = preprocess_text(sentence_df.text)
        con_scores = fa_con.get_fa_scores(df=sentence_df,doc_colname='prep_text')
        try:
            combined_result['concreteness'] = con_scores[foundations_con].to_dict('records')[0]
        except:
            combined_result['concreteness'] = None
    print("returning")
    return combined_result

if __name__ == "__main__":
    CON_Model_PATH = "word2vec-google-news-300"
    model_con = gensim.downloader.load(CON_Model_PATH)
    foundations_con = ['bias_concreteness','intensity_concreteness','concreteness.vice','concreteness.virtue']
    file_path = sys.argv[1]
    data = json.load(file_path)
    result = data['result_list']
    for obj in result:
        for text in obj['text_results']:
            text = MF_concreteness(text, fa_con)

    with open(file_path, 'w', encoding='utf-8') as f:
        # Use NumpyEncoder to convert numpy data to list
        # Previous error: Object of type int64 is not JSON serializable
        json.dump(data, f, indent=4, ensure_ascii=False,
                    cls=NumpyEncoder)
#     print(obj)
