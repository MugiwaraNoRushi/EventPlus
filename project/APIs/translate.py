import json
from collections import Counter
from os import environ
from google.cloud import translate
import pandas as pd
import os

Length_message = 6
# We would need key.json
# Ask Fred if we can put it 

def concattext(textblob):
    # if it's text just return it
    if type(textblob) is str:
        return textblob
    elif type(textblob) is list:
        outstrs = []
        for t in textblob:
            if type(t) is str:
                outstrs.append(t)
            else:
                outstrs.append(t['text'])
        return " ".join(outstrs)


def translate_language(text, target_language_code = "en"):
    try:
        response = client.translate_text(
            contents=[text],
            target_language_code=target_language_code,
            parent=parent,
        )
    except:
        return {"soure_lang": "ERR", "en_text": None}

    for translation in response.translations:
        return {"source_lang": translation.detected_language_code, "en_text": translation.translated_text}

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="key.json"


project_id = "venice-text-translation"
parent = f"projects/{project_id}"
client = translate.TranslationServiceClient()

data = json.load(open("result.json"))

lnct = 0
for midx, m in enumerate(data['messages']):
    if m['type'] == 'message':
        output = concattext(m['text'])
        translation = translate_language(output)
        data['messages'][midx]['translation_en'] = translation

    lnct += 1
    if lnct % 100 == 0:
        print("Lines", lnct)
    if lnct > 200:
        break

# json.dump(data, open("translated.json", "w"))

# Conversion and preprocessing
data_pd = pd.DataFrame(data)

output = data_pd.append(data_pd['messages'],ignore_index = True)

dlist = list(data_pd['messages'])
output = pd.DataFrame(dlist)
# output.info()

newdata = output.dropna(subset = ['translation_en'])

newdata['translation_en'] = newdata['translation_en'].apply(lambda x: x['en_text'])

newdata.rename (columns = {'date':'Date','text':'port_text','translation_en':'text'},inplace = True)

newdata = newdata.dropna(subset = ['text'])

data = newdata[newdata['text'].apply(lambda x: len(x.split()) > Length_message)]

data.to_csv('final_file_to_be_read.csv')
