import pandas as pd
import json

Length_message = sys.argv[2]
file_name = sys.argv[1]

data_pd = pd.read_json(file_name)

output = data_pd.append(data_pd['messages'],ignore_index = True)

dlist = list(data_pd['messages'])
output = pd.DataFrame(dlist)
# output.info()

newdata = output.dropna(subset = ['translation_en'])

newdata['translation_en'] = newdata['translation_en'].apply(lambda x: x['en_text'])

newdata.rename (columns = {'date':'Date','text':'port_text','translation_en':'text'},inplace = True)

newdata = newdata.dropna(subset = ['text'])

data = newdata[newdata['text'].apply(lambda x: len(x.split()) > Length_message)]

file_name = file_name.split('.')[0] + ".csv"
data.to_csv(file_name)
