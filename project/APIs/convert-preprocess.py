import pandas as pd
import json
import sys
# Pass first argument as the input file path and second file path as the output file

Length_message = 6
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

output_file_name = sys.argv[2]
data.to_csv(output_file_name)
