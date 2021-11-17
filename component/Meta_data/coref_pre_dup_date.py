import json
import argparse
import numpy as np
from allennlp.predictors.predictor import Predictor
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.tokens import Doc, Span
import pandas as pd

def get_span_noun_indices(doc, cluster):
    spans = [doc[span[0]:span[1]+1] for span in cluster]
    spans_pos = [[token.pos_ for token in span] for span in spans]
    span_noun_indices = [i for i, span_pos in enumerate(spans_pos)
        if any(pos in span_pos for pos in ['NOUN', 'PROPN'])]
    return span_noun_indices

class NumpyEncoder(json.JSONEncoder):
    """ Custom encoder for numpy data types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
                            np.int16, np.int32, np.int64, np.uint8,
                            np.uint16, np.uint32, np.uint64)):

            return int(obj)

        elif isinstance(obj, (np.float_, np.float16, np.float32, np.float64)):
            return float(obj)

        elif isinstance(obj, (np.complex_, np.complex64, np.complex128)):
            return {'real': obj.real, 'imag': obj.imag}

        elif isinstance(obj, (np.ndarray,)):
            return obj.tolist()

        elif isinstance(obj, (np.bool_)):
            return bool(obj)

        elif isinstance(obj, (np.void)):
            return None

        return json.JSONEncoder.default(self, obj)

def get_coference(doc):
    pred = predictor.predict(document = doc)
    clusters = pred['clusters']
    document = pred['document']
    top_spans = pred['top_spans']

    # find the main span for each cluster
    clusters_top_span = []
    for i in range(0, len(clusters)):
        one_cl = clusters[i]
        span_rank = [top_spans.index(span) for span in one_cl]
        top_span = np.argmin(span_rank)
        clusters_top_span.append(one_cl[top_span])
    pred['clusters_top_span'] = clusters_top_span

    # convert top span for each cluster to text
    clusters_top_span_text = []
    for each_top_span in clusters_top_span:
        span_text = document[each_top_span[0]:(each_top_span[1]+1)]
        clusters_top_span_text.append(span_text)
    pred['clusters_top_span_text'] = clusters_top_span_text


    return pred

def perform_neural_coref(data_pd):
    print("starting neural coreferencing")
    date = data_pd['Date']
    # this will contain the text
    f = data_pd['text']
    docs = [line.rstrip() for line in f]
    # load AllenNLP predictor
    predictor = Predictor.from_path("https://storage.googleapis.com/allennlp-public-models/coref-spanbert-large-2021.03.10.tar.gz")

    for doc_num, doc_text in enumerate(docs):
        # get coreference result for the document
        coref_pred = get_coference(doc_text)
        print('*' * 20)
        print(coref_pred)
        doc_tokens = coref_pred['document']
        doc_nlp = nlp(doc_text)
        # replace coref mention to main mention
        for i_cluster, cluster in enumerate(coref_pred['clusters']):
            print('------')
            print(cluster)
            if get_span_noun_indices(doc_nlp,cluster):
                for mention in cluster:
                    print("--> mention: %s" % mention)
                    # replace each token in the mention range to empty
                    for i in range(mention[0], mention[1]+1):
                        doc_tokens[i] = ''
                    # replace the first token with the main mention
                    doc_tokens[mention[0]] = ' '.join(coref_pred['clusters_top_span_text'][i_cluster])
                    print(doc_tokens)
        docs[doc_num] = ' '.join([i for i in doc_tokens if i])
        print("Replaced docs: %s" % docs[doc_num])

    data_pd['text'] = docs
    return data_pd
