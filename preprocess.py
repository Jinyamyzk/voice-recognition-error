from email import header
import pandas as pd
import re
import random
from pykakasi import kakasi
import urllib
import json
import spacy
from sklearn.model_selection import train_test_split
import glob
from tqdm import tqdm

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

def noise(text):
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

def create_df_for_BERT(df):
    data = []
    for row in tqdm(range(len(df)), desc="[Creating dataframe]"):
        former_idx = row - 5 if row >= 5 else 0
        latter_idx = row + 5 if row + 5 > len(df) else len(df)
        former = "".join(df.iloc[former_idx:row, 2].to_list())
        latter = "".join(df.iloc[row+1:latter_idx, 2].to_list())
        if df.iat[row, 3]: # noisedがあれば
            target = df.iat[row, 3]
            label = 1
        else:
            target = df.iat[row, 2]
            label = 0
        data.append([target, former, latter, label])
    new_df = pd.DataFrame(data, columns=["target", "former", "latter", "label"])

    return new_df

def main():
    conversation = []
    files = glob.glob("btsjcorpus_ver_march_2022_1-29/**/**/*.xlsx")
    for file in tqdm(files, desc="[Loading excel]"):
        df = pd.read_excel(file,index_col=None,names=["speaker","raw_content"],skiprows=[0,1],usecols=[6,7])
        conversation.append(df)
    conversation = pd.concat(conversation, axis=0, ignore_index=True)

    conversation["content"] = conversation["raw_content"].apply(remove_symbol)

    conversation["noised"] = ""
    for row in tqdm(range(len(conversation)), desc="[Noising text]"):
        text = conversation.iloc[row, 2]
        result = noise(text)
        if result:
            conversation.iat[row, 3] = result
    
    df_for_BERT = create_df_for_BERT(conversation)
    
    df_train, df_valid_test = train_test_split(df_for_BERT, test_size=0.2, shuffle=True, random_state=123, stratify=df_for_BERT["label"])
    df_valid, df_test = train_test_split(df_valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=df_valid_test["label"])

    print(f"train: {len(df_train)}, valid: {len(df_valid)}, test: {len(df_test)}")

    df_train.to_csv("data/train.tsv", sep="\t", index=False, header=False)
    df_valid.to_csv("data/valid.tsv", sep="\t", index=False, header=False)
    df_test.to_csv("data/test.tsv", sep="\t", index=False, header=False)
            
if __name__ == "__main__":
    main()