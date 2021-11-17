import json,os,sys
import pandas as pd
import numpy as np
from glob import glob
from datetime import datetime
import pickle as pk
import pandas as pd
from scripts import data_loader


def find_date (message):
    # convert strings of various forms into datetime objects
    date_str = str(message['date']).split(' ')
    date_str[0] = date_str[0].zfill(2)
    date_str = ' '.join(date_str)
    try:
        date_datetime = datetime.strptime(date_str,'%d %b %y')
    except:
        try:
            date_datetime = datetime.strptime(date_str,'%Y-%m-%d')
        except:
            try:
                date_datetime = datetime.strptime(date_str,'%d %b %Y')
            except:
                try:
                    date_datetime = datetime.strptime(date_str,'%d %B %Y')
                except:
                    try:
                        date_datetime = datetime.strptime(date_str,'%d %B %y')
                    except:
                        try:#2020-12-31T23:48:19
                            date_datetime = datetime.strptime(date_str,'%Y-%m-%dT%H:%M:%S')
                        except:
                            date_datetime = datetime(1900,1,1)
                            print('ERROR',date_str)

    return date_datetime

# we extract different things depending on whether data are NEWS or Telegram messages
for data_type in ['Telegram','NEWS']:
    # various places to find data before emotions are added (CHANGE THIS FOR YOUR PIPELINE)
    files = list(glob('/data/input_kg/'+data_type+'/*.json'))
    files += list(glob('/data/before_emotions_added/'+data_type+'/*.json'))

    # ignore data that has been processed
    files = [f for f in files if 'emotions-added_emotions-added.json' not in f]
    # various data to extract from json files
    extracted_data = {'UNIQUE_ID':[],'file':[],'ID':[],'date':[],'event':[],'text':[]}
    unique_reports=  []
    # unorganized set of unique dates
    dates=set()
    # moral foundations
    moral_foundations = ['authority.virtue', 'authority.vice', 'fairness.virtue', 'fairness.vice', 'general_morality.virtue', 'general_morality.vice', 'harm.virtue', 'harm.vice', 'ingroup.vice', 'ingroup.virtue', 'purity.vice', 'purity.virtue']
    # concreteness features
    concreteness_foundations = ['bias_concreteness', 'intensity_concreteness', 'concreteness.vice', 'concreteness.virtue']
    # emotion columns. these columns are needed to make SpanEmo work
    emotion_cols = ['ID','Tweet','anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']
    emotion_text = {key:[] for key in emotion_cols}
    # for all json files
    for file in files:
        # see what file is being parsed
        print(file)
        # load file
        data=json.load(open(file,'rb'))
        # for each message ID, message extracted
        for ID,message in enumerate(data['result_list']):
            # ignore if no data
            if message['text_results'] == []: continue

            lengths = np.array([len(extracted_data[key]) for key in extracted_data.keys()])
            # ID as an INT
            ID = int(float(ID))
            # unique IDs (in cases data are duplicates)
            if str(message) not in unique_reports:
                unique_reports.append(str(message))
            extracted_data['UNIQUE_ID'].append(unique_reports.index(str(message)))
            # record file
            extracted_data['file'].append(file)
            # record ID
            extracted_data['ID'].append(ID)
            # record message/article date
            dates.add(find_date(message))
            extracted_data['date'].append(date_datetime)
            message_features = {}
            event = False

            sentence_text = []
            for sentence in message['text_results']:
                # raw text
                s = str(sentence['sentence'])
                sentence_text.append(s)
                # record event (for my own work)
                if 'events' in sentence.keys():
                    if len(sentence['events']) > 0:
                        if sentence['events'][0]['event_type'] != '':
                            event = True
                # moral foundations
                mf = None
                if 'mf' in sentence.keys():
                    mf = sentence['mf']
                else:
                    for moral_foundation in moral_foundations:
                        message_features[moral_foundation].append(None)
                for moral_foundation in moral_foundations:
                    if moral_foundation not in extracted_data.keys():
                        extracted_data[moral_foundation] = []
                    if moral_foundation not in message_features.keys():
                        message_features[moral_foundation] = []
                    if mf is None or 'mf' not in sentence.keys():
                        message_features[moral_foundation].append(None)
                    else:
                        message_features[moral_foundation].append((mf[moral_foundation]))
                # concreteness data
                concreteness = None
                if 'concreteness' in sentence.keys():
                    concreteness = sentence['concreteness']
                for concreteness_foundation in concreteness_foundations:
                    if concreteness_foundation not in extracted_data.keys():
                        extracted_data[concreteness_foundation] = []
                    if concreteness_foundation not in message_features.keys():
                        message_features[concreteness_foundation] = []
                    if concreteness is None or  'concreteness' not in sentence.keys():
                        message_features[concreteness_foundation].append(None)
                    else:
                        message_features[concreteness_foundation].append((concreteness[concreteness_foundation]))
            if data_type == 'Telegram': # extract whole messages for telegram emotion
                sentence_text = ' '.join(sentence_text)
                emotion_text['ID'].append(unique_reports.index(str(message)))
                emotion_text['Tweet'].append(sentence_text)

            elif data_type == 'NEWS': # extract sentences for news emotion
                for j,s in enumerate(sentence_text):
                    emotion_text['ID'].append(str(unique_reports.index(str(message))) + '_'+str(j))
                    emotion_text['Tweet'].append(s)

                sentence_text = ' '.join(sentence_text)
            extracted_data['text'].append(sentence_text)
            mean_features = {key: np.mean([val for val in message_features[key] if val is not None]) for key in message_features.keys()}
            for key in mean_features.keys():
                extracted_data[key].append(mean_features[key])
            extracted_data['event'].append(event)
            for key in emotion_text.keys():
                if key not in 'ID' and key not in 'Tweet':
                    emotion_text[key] = [0]*len(emotion_text['ID'])

    # use this file when creating emotions
    unfilled_emotion_file = 'emotion_text_'+data_type+'.csv'
    if len(files) == 0:
        # record bad data
        unfiltered_data=pd.read_csv(unfilled_emotion_file,sep='\t')
        preprocessor = data_loader.twitter_preprocessor()
        bad_data = [False]*len(unfiltered_data)
        for ii,[n,row] in enumerate(unfiltered_data.iterrows()):
            try:
                test = ' '.join(preprocessor(row['Tweet']))
            except:
                test = 'NONE'
                bad_data[ii] = True
                print([n,row['Tweet']])


        unfiltered_data.loc[bad_data,'Tweet'] = 'None'
        unfiltered_data["Tweet"] = unfiltered_data["Tweet"].astype(str)
        unfiltered_data.to_csv(unfilled_emotion_file,sep='\t',index=False)
    else: # else put emotions back into JSON file
        pd.DataFrame(emotion_text).to_csv(unfilled_emotion_file,index=False,sep='\t')
        print('unfilled emots made')
        filled_emotion_file = unfilled_emotion_file[:-4]+'_filled.csv'
        if os.path.exists(filled_emotion_file):
            emotions = ['anger','anticipation','disgust','fear','joy','love','optimism','pessimism','sadness','surprise','trust']
            emotion_data = pd.read_csv(filled_emotion_file).astype(str)
            emotion_data = emotion_data.rename(columns={'pred_'+str(i):emot for i,emot in enumerate(emotions)})

            unfilled_emotion_data = pd.read_csv(unfilled_emotion_file,sep='\t')
            # fill emotions
            emotion_data['message_ID'] = [f if type(f) == int else f.split('_')[0] for f in unfilled_emotion_data['ID']]
            if type(unfilled_emotion_data['ID'][0]) is str:
                emotion_data['line'] = [f.split('_')[1] for f in unfilled_emotion_data['ID']]
            emotion_data['message_ID'] = emotion_data['message_ID'].astype(int)
            if 'line' in emotion_data.columns:
                emotion_data['line'] = emotion_data['line'].astype(int)

            df_extracted = pd.DataFrame(extracted_data)
            mean_emotions = emotion_data
            # if emotions are split by sentences
            if len(emotion_data) > len(df_extracted):
                mean_emotions = {emot:[] for emot in emotion_data.columns}
                # data by unique ID
                for ID in df_extracted['UNIQUE_ID']:
                     emots = emotion_data.loc[emotion_data['message_ID']==ID,]
                     for emot in emots.columns:
                         mean = np.mean(emots[emot].values.astype(float))
                         mean_emotions[emot].append(mean)
                mean_emotions = pd.DataFrame(mean_emotions)

            extracted_data = pd.concat([pd.DataFrame(extracted_data),mean_emotions],axis=1).astype(str)
            extracted_data['UNIQUE_ID'] = extracted_data['UNIQUE_ID'].astype(int)
            # put emotions into json file here
            emotion_data['file'] = [extracted_data.loc[extracted_data['UNIQUE_ID']==ID,'file'].values[0] for ID in emotion_data['message_ID']]
            emotion_data['ID'] = [extracted_data.loc[extracted_data['UNIQUE_ID']==ID,'ID'].values[0] for ID in emotion_data['message_ID']]
            unique_files = set(list(emotion_data['file'].values))
            kg_emotions = {file:json.load(open(file,'rb')) for file in unique_files}
            if 'ID' in emotion_data.columns:
                emotion_data['ID'] = emotion_data['ID'].astype(int)
            if 'line' in emotion_data.columns:
                emotion_data['line'] = emotion_data['line'].astype(int)
            for n,row in emotion_data.iterrows():
                file = row['file']
                ID = row['ID']
                # dict of emotion: binary (sentence/message does or does not have emotion)
                message_emotions = {emot:float(row[emot]) for emot in emotions}

                if 'line' not in emotion_data.columns:#'_' not in ID:
                    kg_emotions[file]['result_list'][ID]['emotions']=message_emotions
                else:
                    line = row['line']

                    kg_emotions[file]['result_list'][ID]['text_results'][line]['emotions'] = message_emotions
            for file in unique_files:
                # replace file with new emotions-added file
                outfile = file[:-5]+'_emotions-added.json'
                if file in glob('/data/before_emotions_added/'+data_type+'/*.json'):
                    outfile = '/data/input_kg/'+data_type+'/'+file.split('/')[-1][:-5]+'_emotions-added.json'
                if 'emotions-added_emotions-added.json' not in outfile:
                    json.dump(kg_emotions[file],open(outfile,'w',encoding="utf8"))

        pd.DataFrame(extracted_data).to_csv('extracted_telegram_features_'+data_type+'.csv',index=False)
