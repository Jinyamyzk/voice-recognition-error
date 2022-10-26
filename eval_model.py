import torch
from torchtext.legacy import data
from torch import nn
from transformers import BertJapaneseTokenizer
from utils.setence_bert_classifier import SentenceBertClassifier
import pandas as pd
import csv
from tqdm import tqdm

# 日本語BERTの分かち書き用tokenizerです
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
max_length = 512  # 東北大学_日本語版の最大の単語数（サブワード数）は512

def get_text(batch_ids, ref):
    batch_ids = batch_ids.tolist()

    text_ids = []
    former_text_ids = []
    latter_text_ids = []

    for id in batch_ids:
        text = tokenizer_512(ref.iat[id, 1])
        former = tokenizer_512("".join(ref.iloc[id-5:id, 0].to_list()))
        latter = tokenizer_512("".join(ref.iloc[id+1:id+5, 0].to_list()))
        text_ids.append(text)
        former_text_ids.append(former)
        latter_text_ids.append(latter)
    return torch.tensor(text_ids), torch.tensor(former_text_ids), torch.tensor(latter_text_ids)

def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, truncation=True, padding="max_length" ,max_length=512)

def main():
    # テストデータでの正解率を求める
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    ref = pd.read_csv("data/ref/data_ref.tsv", sep="\t", index_col=0)

    ID = data.Field(sequential=False, use_vocab=False)
    LABEL = data.Field(sequential=False, use_vocab=False)

    dataset_train, dataset_valid, dataset_test = data.TabularDataset.splits(
        path="data", train="train.tsv", validation="valid.tsv",test="test.tsv", format="tsv", fields=[
            ("Id", ID), ("Label", LABEL)])
    
    # DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
    batch_size = 16  # BERTでは16、32あたりを使用する

    dl_test = data.Iterator(
        dataset_test, batch_size=batch_size, train=False, sort=False)
    
    model = SentenceBertClassifier()
    state_dict = torch.load("model/model_trained.pt")
    model.load_state_dict(state_dict)

    # 損失関数の設定
    criterion = nn.CrossEntropyLoss()

    model.eval()   # モデルを検証モードに
    model.to(device)  # GPUが使えるならGPUへ送る

    # epochの正解数を記録する変数
    epoch_corrects = 0
    test_result = []
    for batch in tqdm(dl_test):  # testデータのDataLoader
        text_ids, former_text_ids, latter_text_ids = get_text(batch.Id, ref)
        text_ids = text_ids.to(device)
        former_text_ids = former_text_ids.to(device)
        latter_text_ids = latter_text_ids.to(device)
        labels = batch.Label.to(device)  # ラベル

        # 順伝搬（forward）計算
        with torch.set_grad_enabled(False):

            outputs = model(text_ids, former_text_ids, latter_text_ids)

            loss = criterion(outputs, labels)  # 損失を計算
            _, preds = torch.max(outputs, 1)  # ラベルを予測
            epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

            for text, former, latter, correct in zip(text_ids.tolist(), former_text_ids.tolist(), latter_text_ids.tolist(), (preds == labels.data).tolist()):
                text = "".join(tokenizer.convert_ids_to_tokens(text))
                former = "".join(tokenizer.convert_ids_to_tokens(former))
                latter = "".join(tokenizer.convert_ids_to_tokens(latter))
                test_result.append([
                    text, former, latter, correct
                ])



    # 正解率
    epoch_acc = epoch_corrects.double() / len(dl_test.dataset)

    print('テストデータ{}個での正解率：{:.4f}'.format(len(dl_test.dataset), epoch_acc))

    # 結果の書き出し
    with open('test_result.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(test_result)

if __name__ == "__main__":
    main()