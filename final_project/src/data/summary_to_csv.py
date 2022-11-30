import pandas as pd
import json
import os
from pathlib import Path

# gets summary of all raw datasets


if __name__ == '__main__':

    dict_lst = []

    with open('summary.txt') as f:
        for line in f.readlines():
            curr_line = line
            curr_line = curr_line.replace('.narrowPeak.gz', ';')
            curr_line = curr_line.replace('\t', '')
            curr_line = curr_line.replace('\n', '')
            curr_line = curr_line.replace(' ', '')
            curr_line = curr_line.replace('=', '\":\"')
            curr_line = curr_line.replace(';', '\",\"')
            curr_line = str("{ \"ID\":\"") + curr_line + str("\"}")
            curr_line_dict = json.loads(curr_line)
            dict_lst.append(curr_line_dict)

    # path = '/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data'
    path = Path(os.getcwd())
    path = f'{str(path.parent.parent)}/data/'
    if not os.path.exists(path):
        os.mkdir(path)

    df = pd.DataFrame.from_dict(dict_lst)
    csv = df.to_csv(f'{path}/summary.csv')
