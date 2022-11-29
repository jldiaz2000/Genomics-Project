
import re
import requests
from bs4 import BeautifulSoup
import os


# scrapes dataset from url below

url = 'http://cnn.csail.mit.edu/motif_discovery/'
html_text = requests.get(url).text
soup = BeautifulSoup(html_text, 'html.parser')


def download_file(curr_folder, curr_url):
    # Get the title of the track from the HTML element
    train_url = f'{curr_url}train.data'
    test_url = f'{curr_url}test.data'

    path = f'/Users/Jacob_Diaz/Desktop/Cornell/Fall 2022/CS 4775/final-project/data/raw/{curr_folder}'

    if not os.path.exists(path):
        os.mkdir(path)

    # Download the track
    r = requests.get(train_url, allow_redirects=True)
    with open(f'{path}/train.txt', 'wb') as f:
        f.write(r.content)

    r = requests.get(test_url, allow_redirects=True)
    with open(f'{path}/test.txt', 'wb') as f:
        f.write(r.content)


if __name__ == '__main__':

    folders = soup.find_all('a')

    attrs = {'href': re.compile(r'\.data$')}
    for i in range(5, len(folders)):

        curr_folder = folders[i]['href']
        print(f'{i}: {curr_folder}')

        curr_url = f'{url}/{curr_folder}'
        curr_html_text = requests.get(curr_url).text
        curr_sub_folders = BeautifulSoup(
            curr_html_text, 'html.parser').find_all('a', attrs=attrs)
        # print(curr_sub_folders)
        # print(list(map(lambda x: x.text.strip(), curr_sub_folders)))
        str_lst = list(map(lambda x: x.text.strip(), curr_sub_folders))
        # assert(len(str_lst) == 2)
        # assert('test.data' in str_lst)
        # assert('train.data' in str_lst)

        download_file(curr_folder, curr_url)
