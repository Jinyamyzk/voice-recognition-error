from os import sep
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm

def main():
    df_1 = pd.read_csv("noised/noised_1.tsv", sep="\t")
    df_2 = pd.read_csv("noised/noised_2.tsv", sep="\t")
    df = pd.concat([df_1, df_2])
    df = df[["content", "noised"]]

    # label列を追加
    df.loc[df["noised"].isna(), "label"] = 0
    df.loc[~df["noised"].isna(), "label"] = 1
    df["label"] = df["label"].astype(int)


    # noisedのないところをcontentで埋める
    df["noised"].where(df["label"]==1, df["content"],  inplace=True)

    # id列を追加
    df["id"] = list(range(len(df)))

    # noisedのあるid
    ids_noised = df[df['label']==1][["id","label"]]
    # noisedのないidを同じだけ取得
    ids_not_noised = df[df["label"]==0][["id", "label"]].sample(n=len(ids_noised))
    # 結合
    ids = pd.concat([ids_noised, ids_not_noised])
    print(f"num noised: {len(ids_noised)}, num not noised: {len(ids_not_noised)}")

    dt_train, df_valid_test = train_test_split(ids, test_size=0.2, shuffle=True, random_state=123, stratify=ids["label"])
    df_valid, dt_fest = train_test_split(df_valid_test, test_size=0.5, shuffle=True, random_state=123, stratify=df_valid_test["label"])

    print(f"train: {len(dt_train)}, valid: {len(df_valid)}, test: {len(dt_fest)}")

    df.to_csv("data/ref/data_ref", sep="\t")

    dt_train.to_csv("data/train.tsv", sep="\t", index=False, header=False)
    df_valid.to_csv("data/valid.tsv", sep="\t", index=False, header=False)
    dt_fest.to_csv("data/test.tsv", sep="\t", index=False, header=False)

if __name__ == "__main__":
    main()
