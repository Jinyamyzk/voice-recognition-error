import torch
from torchtext.legacy import data
import torch.optim as optim
from torch import nn
from transformers import BertModel
from transformers import BertJapaneseTokenizer
from utils.setence_bert_classifier import SentenceBertClassifier
import matplotlib.pyplot as plt
from tqdm import tqdm


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

    train_loss_list = []
    train_acc_list = []
    valid_loss_list = []
    valid_acc_list = []

     # epochのループ
    for epoch in range(num_epochs):
        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
            else:
                net.eval()   # モデルを検証モードに

            epoch_loss = 0.0  # epochの損失和
            epoch_corrects = 0  # epochの正解数
            iteration = 1

            # データローダーからミニバッチを取り出すループ
            for batch in (dataloaders_dict[phase]):
                # batchはTextとLableの辞書型変数

                # GPUが使えるならGPUにデータを送る
                text_ids = batch.Text[0].to(device)
                former_text_ids = batch.FormerText[0].to(device)
                latter_text_ids = batch.LatterText[0].to(device)
                labels = batch.Label.to(device)  # ラベル

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):

                    # BERTに入力
                    outputs = net(text_ids, former_text_ids, latter_text_ids)

                    loss = criterion(outputs, labels)  # 損失を計算

                    _, preds = torch.max(outputs, 1)  # ラベルを予測

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            acc = (torch.sum(preds == labels.data)
                                   ).double()/batch_size
                            print('イテレーション {} || Loss: {:.4f} || 10iter. || 本イテレーションの正解率：{}'.format(
                                iteration, loss.item(),  acc))

                    iteration += 1

                    # 損失と正解数の合計を更新
                    epoch_loss += loss.item() * batch_size
                    epoch_corrects += torch.sum(preds == labels.data)

            # epochごとのlossと正解率
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)

            print('Epoch {}/{} | {:^5} |  Loss: {:.4f} Acc: {:.4f}'.format(epoch+1, num_epochs,
                                                                           phase, epoch_loss, epoch_acc))
            
            epoch_acc.cpu()
            if phase == "train":
                train_loss_list.append(epoch_loss)
                train_acc_list.append(epoch_acc)
            else:
                valid_loss_list.append(epoch_loss)
                valid_acc_list.append(epoch_acc)
    
    # 学習曲線 (損失関数値)
    plt.figure(figsize=(8,6))
    plt.plot(valid_loss_list,label='adam', lw=3, c='b')
    plt.title('learning curve (loss)')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.savefig("loss.png")

    # 学習曲線 (精度)
    plt.figure(figsize=(8,6))
    plt.plot(valid_acc_list,label='adam', lw=3, c='b')
    plt.title('learning curve (accuracy)')
    plt.xticks(size=14)
    plt.yticks(size=14)
    plt.grid(lw=2)
    plt.legend(fontsize=14)
    plt.savefig("acc.png")

    return net



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

# 訓練モードに設定
net.train()

# 勾配計算を分類アダプターのみ実行
# 1. まず全てを、勾配計算Falseにしてしまう
for param in net.parameters():
    param.requires_grad = False
# 2. 識別器を勾配計算ありに変更
for param in net.cls.parameters():
    param.requires_grad = True

# 最適化手法の設定
optimizer = optim.Adam([{'params': net.cls.parameters(), 'lr': 1e-4}])
# 損失関数の設定
criterion = nn.CrossEntropyLoss()

# 学習・検証を実行する。1epochに2分ほどかかります
num_epochs = 1
net_trained = train_model(net, dataloaders_dict,
                          criterion, optimizer, num_epochs=num_epochs)

# モデルの保存
torch.save(net_trained.state_dict(), "model")




# テストデータでの正解率を求める
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

net_trained.eval()   # モデルを検証モードに
net_trained.to(device)  # GPUが使えるならGPUへ送る

# epochの正解数を記録する変数
epoch_corrects = 0

for batch in tqdm(dl_test):  # testデータのDataLoader
    # batchはTextとLableの辞書オブジェクト
    # GPUが使えるならGPUにデータを送る
    text_ids = batch.Text[0].to(device)
    former_text_ids = batch.FormerText[0].to(device)
    latter_text_ids = batch.LatterText[0].to(device)
    labels = batch.Label.to(device)  # ラベル

    # 順伝搬（forward）計算
    with torch.set_grad_enabled(False):

        # BertForLivedoorに入力
        outputs = net_trained(text_ids, former_text_ids, latter_text_ids)

        loss = criterion(outputs, labels)  # 損失を計算
        _, preds = torch.max(outputs, 1)  # ラベルを予測
        epoch_corrects += torch.sum(preds == labels.data)  # 正解数の合計を更新

# 正解率
epoch_acc = epoch_corrects.double() / len(dl_test.dataset)

print('テストデータ{}個での正解率：{:.4f}'.format(len(dl_test.dataset), epoch_acc))






