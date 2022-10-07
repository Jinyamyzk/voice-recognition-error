import torch
from torchtext.legacy import data
from transformers import BertModel
from transformers import BertJapaneseTokenizer
from utils.setence_bert_classifier import SentenceBertClassifier


# モデルを学習させる関数を作成
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)
    print('-----start-------')

    # ネットワークをGPUへ
    net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # ミニバッチのサイズ
    batch_size = dataloaders_dict["train"].batch_size



# 日本語BERTの分かち書き用tokenizerです
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")

max_length = 512  # 東北大学_日本語版の最大の単語数（サブワード数）は512

def tokenizer_512(input_text):
    """torchtextのtokenizerとして扱えるように、512単語のpytorchでのencodeを定義。ここで[0]を指定し忘れないように"""
    return tokenizer.encode(input_text, truncation=True, max_length=512, return_tensors="pt")[0]

TEXT = data.Field(sequential=True, tokenize=tokenizer_512, use_vocab=False, lower=False,
                            include_lengths=True, batch_first=True, fix_length=max_length, pad_token=0)           
LABEL = data.Field(sequential=False, use_vocab=False)

dataset_train, dataset_valid, dataset_test = data.TabularDataset.splits(
    path="data", train="train.tsv", validation="valid.tsv",test="test.tsv", format="tsv", fields=[
        ("Text", TEXT), ("FormerText", TEXT), ("LatterText", TEXT), ("Label", LABEL)])

item = next(iter(dataset_train))
print(item.Text)
print("長さ：", len(item.Text))  # 長さを確認 [CLS]から始まり[SEP]で終わる。512より長いと後ろが切れる
print("ラベル：", item.Label)

print(tokenizer.convert_ids_to_tokens(item.Text.tolist())) 
print(tokenizer.convert_ids_to_tokens(item.FormerText.tolist())) 
print(tokenizer.convert_ids_to_tokens(item.LatterText.tolist())) 

# DataLoaderを作成します（torchtextの文脈では単純にiteraterと呼ばれています）
batch_size = 16  # BERTでは16、32あたりを使用する

dl_train = data.Iterator(
    dataset_train, batch_size=batch_size, train=True)

dl_valid = data.Iterator(
    dataset_valid, batch_size=batch_size, train=False, sort=False)

dl_test = data.Iterator(
    dataset_test, batch_size=batch_size, train=False, sort=False)

# 辞書オブジェクトにまとめる
dataloaders_dict = {"train": dl_train, "val": dl_valid}

batch = next(iter(dl_test))
print(batch)
print(batch.Text[0].shape)
print(batch.Label.shape)


net = SentenceBertClassifier()

# データローダーからミニバッチを取り出すループ
for batch in (dataloaders_dict["train"]):
    # batchはTextとLableの辞書型変数

    # GPUが使えるならGPUにデータを送る
    text_ids = batch.Text[0]
    former_text_ids = batch.FormerText[0]
    latter_text_ids = batch.LatterText[0]
    labels = batch.Label.size()  # ラベル
    outputs = net(text_ids, former_text_ids, latter_text_ids)
    print(outputs.size())