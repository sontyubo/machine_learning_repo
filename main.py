import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def main():
    wine_dataset : Bunch = load_wine()   # type: ignore

    exp = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
    target = pd.DataFrame(wine_dataset.target, columns=['target'])

    df = pd.concat([exp, target], axis=1)
    # print(df)

    # 説明変数
    X = df[wine_dataset.feature_names]
    # print(X)
    
    # 目的変数
    y = df['target']
    # print(y)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
    
    # 標準化スケーラー
    scaler = StandardScaler()

    # 説明変数の訓練用データを標準化
    X_train_scaled = scaler.fit_transform(X_train)

    # ロジスティック回帰モデル（分類モデル）
    model = LogisticRegression()

    # モデルの学習
    model.fit(X_train_scaled, y_train)

    # 説明変数のテスト用データを標準化
    X_test_scaled = scaler.transform(X_test)

    # 学習済みモデルを使い、テスト用データで予測を行う
    y_pred = model.predict(X_test_scaled)

    # 正解率を計算
    print()
    print("正解割合は...")
    print(accuracy_score(y_test, y_pred))


if __name__ == "__main__":
    main()
