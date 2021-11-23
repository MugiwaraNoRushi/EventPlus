import json
from collections import Counter
from os import environ
from google.cloud import translate
import pandas as pd
import os


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

file_name = sys.argv[1]
data = json.load(open(file_name))

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

output_file_name = file_name.split(".") + "translated" + ".json"
json.dump(data, open(output_file_name, "w"))

# Conversion and preprocessing
