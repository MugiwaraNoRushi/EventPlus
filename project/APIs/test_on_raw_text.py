# pass the first argument as the event json
import pandas as pd
import sys
import os
import json
import re
import nltk
from nltk import tokenize
import numpy as np
from frameAxis import FrameAxis
from gensim.models import KeyedVectors
import gensim
import gensim.downloader
sys.path.append("..")
sys.path.append("../component/Meta_data")
from component.Meta_data.coref_pre_dup_date import perform_neural_coref

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

def save(args, result_list, not_done_list):
    with open(args.save_path, 'wb') as f:
        pickle.dump(result_list, f)

    result_json = {
        'error_list': not_done_list,
        'result_list': result_list
    }
    with open(args.save_path_json, 'w', encoding='utf-8') as f:
        # Use NumpyEncoder to convert numpy data to list
        # Previous error: Object of type int64 is not JSON serializable
        json.dump(result_json, f, indent=4, ensure_ascii=False,
                    cls=NumpyEncoder)
    print ('Saved')

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

def MF_concreteness(combined_result,fa_mf, fa_con):
    if len(combined_result["tokens"]) >=5:
        sentence_df = pd.DataFrame([combined_result['sentence']], columns=['text'])
        sentence_df['prep_text'] = preprocess_text(sentence_df.text)
        mf_scores = fa_mf.get_fa_scores(df=sentence_df, doc_colname='prep_text', tfidf=False, format="virtue_vice")
        con_scores = fa_con.get_fa_scores(df=sentence_df,doc_colname='prep_text')
        try:
            combined_result['mf'] = mf_scores[foundations_mf].to_dict(orient='records')[0]
        except KeyError:
            #print(mf_scores)
            combined_result['mf'] = None
        try:
            combined_result['concreteness'] = con_scores[foundations_con].to_dict('records')[0]
        except:
            combined_result['concreteness'] = None
    print("returning")
    return combined_result

if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('-data', type=str, default='../../raw_text/julsepscan.txt')
    # p.add_argument('-data', type=str, default='../../ace_data/ace_rawtext_test.pkl')
    p.add_argument('-save_path', type=str, default='../../raw_text/julsepscan_pipelined.pkl')
    p.add_argument('-save_path_json', type=str, default='../../raw_text/julsepscan_pipelined.json')
    p.add_argument('-negation_detection', action='store_true', default=True,
                    help='Whether detection negation cue and scope resolution')
    # Adding the new arguments for the telegram group id and Moral Foundation model
    p.add_argument('-telegram_group_id',type = str,default = '',help = ' choose the group id for telegram group')
    p.add_argument('-type_of_mf_model',default = False,help ='whether to use the second model for MF or not')
    args = p.parse_args()

    if args.negation_detection:
        eventAPIs = EventAPIs(negation_detection=True)
    else:
        eventAPIs = EventAPIs(negation_detection=False)
    print ('Loaded class')

# --------------Loading all the models---------------------

    print("Running FrameAxis Moral Foundations scores")
    CON_Model_PATH = "word2vec-google-news-300"
    model_con = gensim.downloader.load(CON_Model_PATH)
    foundations_con = ['bias_concreteness','intensity_concreteness','concreteness.vice','concreteness.virtue']
    fa_con = FrameAxis(mfd="customized", w2v_model=model_con)
    if args.type_of_mf_model:
        print("BERT Moral Foundation loading")
        pass
    else:
        print("FrameAxis Model loading")
        MF_Model_PATH = "../component/Embedding_MF/w2v_aylien_huff.txt"
        model_mf = KeyedVectors.load_word2vec_format(MF_Model_PATH, binary=False)
        fa_mf = FrameAxis(mfd="mfd", w2v_model=model_mf)
        foundations_mf = ['authority.virtue', 'authority.vice',
           'fairness.virtue', 'fairness.vice', 'general_morality.virtue',
           'general_morality.vice', 'harm.virtue', 'harm.vice', 'ingroup.vice',
           'ingroup.virtue', 'purity.vice', 'purity.virtue']

    print("MF models loaded")

    print("Frame Axis Models loaded")

# ---------------Loading models ends ends-----------------------------

    not_done_list = []
    data_pd = pd.read_csv(args.data,error_bad_lines=False)
    final_dict_result = data_pd[['id','Date','from','from_id']].to_dict('records')
    TELEGRAM_GROUP_ID = args.telegram_group_id
    f = data_pd['text']
    linelist = [line.rstrip() for line in f]
    data = []
    total_sen = 0
    # convert row text to list of sentences
    for line in linelist:
        sen_list = []
        if line != '':
            # divide to sentences
            sen_list = tokenize.sent_tokenize(line)
        data.append(sen_list)
        total_sen += len(sen_list)
    print ('Total sentences: ', total_sen)
    print ('Total lines: ', len(data))
    result_list = []

    for i_line, sen_list in enumerate(data):
        # this contains the original output
        result_list_this_line = []
        final_output = final_data_result[i_line]
        final_output['group_id'] = TELEGRAM_GROUP_ID
        final_output['text_results'] = result_list_this_line
        for i_sen, text in enumerate(sen_list):
            print ('='*40, 'line num: ', i_line, "; sen num: ", i_sen)
            params_this = {
                'text': text,
                'domain': 'news'
            }
            try:
                combined_result = eventAPIs.analyze(params_this)
                combined_result['line_num'] = i_line
                combined_result['sen_num'] = i_sen
                combined_result['sentence'] = text
                # Adding MF Scores !!
                combined_result = MF_concreteness(combined_result, fa_mf, fa_con)
                print("after returning",combined_result)
                # MF Scores ends !!
                result_list_this_line.append(combined_result)
            except Exception as e:
                print('?'*60)
                print('Error for this text: ', text)
                print(str(e))
                not_done_list.append([i_line, i_sen])
        # result_list.append(result_list_this_line)
        result_list.append(final_output)
        if i_line % 20 == 0:
            save(args, result_list, not_done_list)

    # print (result_list)
    save(args, result_list, not_done_list)
    print ('Not successfuly text:')
    print (not_done_list)
