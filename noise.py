import pandas as pd
import re
import random
from pykakasi import kakasi
import urllib
import json
import spacy
from sklearn.model_selection import train_test_split
import glob
import os
import argparse
from tqdm import tqdm

import warnings
warnings.simplefilter("ignore") # tqdmの出力を見やすくする

kakasi = kakasi()
nlp = spacy.load('ja_ginza')

def remove_symbol(text):
    text = re.sub("\[","(",text)
    text = re.sub("\]",")",text)
    pattern = "\(.*?\)|\<.*?\>|《.*?》|\{.*?\}|【|】|#|'|…|“|”|="
    text = re.sub(pattern, "", text)
    return text

def ginza_tokenize(text):
    doc = nlp(text)
    tokenized = [[token.orth_ for token in sent] for sent in doc.sents][0]
    return tokenized

def kanji_to_hiragana(word):
    result = kakasi.convert(word)
    return result[0]["hira"]

def hiragana_to_kanji(word_yomi):
    url = "http://www.google.com/transliterate?"
    param = {'langpair':'ja-Hira|ja','text':word_yomi}
    paramStr = urllib.parse.urlencode(param)
    readObj = urllib.request.urlopen(url + paramStr)
    response = readObj.read()
    data = json.loads(response)
    fixed_data = json.loads(json.dumps(data[0], ensure_ascii=False))
    return fixed_data[1]

def is_not_kanji(word):
    not_kanji = re.compile(r'[あ-ん]+|[ア-ン]+|[ｱ-ﾝ]+|[0-9０-９]+')
    return not_kanji.search(word)

def noise(text, pbar):
    random.seed(0)
    pbar.update(1)
    tokenized = ginza_tokenize(text)
    random_order = list(range(len(tokenized)))
    random.shuffle(random_order)
    for idx in random_order:
        target = tokenized[idx]
        yomi = kanji_to_hiragana(target)
        if len(yomi) <3:
            continue
        kanji_candidates = hiragana_to_kanji(yomi)
        noised = ""
        for kanji in kanji_candidates:
            if is_not_kanji(kanji):
                break
            if kanji != target:
                noised = kanji
                print(f'{target}: {noised}')
                tokenized[idx] = noised
                return "".join(tokenized)
    return  ""


def noise_(text):
    random.seed(0)
    tokenized = ginza_tokenize(text)
    random_order = list(range(len(tokenized)))
    random.shuffle(random_order)
    for idx in random_order:
        target = tokenized[idx]
        yomi = kanji_to_hiragana(target)
        if len(yomi) <3:
            continue
        kanji_candidates = hiragana_to_kanji(yomi)
        noised = ""
        for kanji in kanji_candidates:
            if is_not_kanji(kanji):
                break
            if kanji != target:
                noised = kanji
                print(f'{target}: {noised}')
                tokenized[idx] = noised
                return "".join(tokenized)
    return  ""

def main(folder_num):
    # # 既にtrain.tsvが存在している場合中止
    # if os.path.isfile("data/train.tsv"):
    #     raise FileExistsError("data/train.tsv already exists")
    os.makedirs("noised", exist_ok=True)

    if folder_num == "t":
        files = glob.glob("test/**/**/*.xlsx")
    else:        
        files = glob.glob(f"btsjcorpus_ver_march_2022_1-29_{folder_num}/**/**/*.xlsx")

    conversation_list = []
    for file in tqdm(files, desc="[Loading excel]"):
        df = pd.read_excel(file,index_col=None,names=["speaker","raw_content"],skiprows=[0,1],usecols=[6,7])
        conversation_list.append(df)
    conversation = pd.concat(conversation_list, axis=0)    
    print(f"データ\n{conversation.info()}")
    conversation["content"] = conversation["raw_content"].apply(remove_symbol)
    conversation["noised"] = ""
    for row in tqdm(range(len(conversation)), desc="[Noising texts]"):
        text = conversation.iloc[row, 2]
        result = noise_(text)
        if result:
            conversation.iat[row, 3] = result
    
    conversation.to_csv(f"noised/noised_{folder_num}.tsv", sep="\t")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("folder_num", help="Choose folder (1,2, or t for test)")
    args = parser.parse_args()
    main(args.folder_num)