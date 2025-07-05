import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import load_wine
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import Bunch


def show_inference(test_size: float) -> list[float]:
    wine_dataset: Bunch = load_wine()  # type: ignore

    exp = pd.DataFrame(wine_dataset.data, columns=wine_dataset.feature_names)
    target = pd.DataFrame(wine_dataset.target, columns=['target'])

    df = pd.concat([exp, target], axis=1)

    X = df[wine_dataset.feature_names]
    y = df['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=0)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    y_train_pred = model.predict(X_train_scaled)

    accuracy = float(accuracy_score(y_test, y_pred))
    accuracy_train = float(accuracy_score(y_train, y_train_pred))

    print()
    print(f"トレーニングデータ:{(1 - test_size) * 100:.1f}%, テストデータ{test_size * 100:.1f}%")
    print("正解割合は...")
    print(accuracy)

    return [accuracy, accuracy_train]


def main():
    test_sizes = [num * 0.05 for num in range(19, 0, -1)]
    train_sizes = [(1 - ts) * 100 for ts in test_sizes]  # トレーニングデータの割合 (%)

    test_accuracies = []
    train_accuracies = []

    for ts in test_sizes:
        acc_test, acc_train = show_inference(ts)
        test_accuracies.append(acc_test)
        train_accuracies.append(acc_train)

    # --- 学習曲線の描画 ---
    plt.figure(figsize=(10, 6))
    plt.plot(train_sizes, train_accuracies, marker='o', label='Train Accuracy')
    plt.plot(train_sizes, test_accuracies, marker='s', label='Test Accuracy')
    plt.title('Learning Curve (Logistic Regression on Wine Dataset)')
    plt.xlabel('Training Data Size (%)')
    plt.ylabel('Accuracy')
    plt.ylim(0.75, 1.0)
    plt.xticks(range(0, 101, 5))
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("./plot.png") 
    plt.close()

    # for local environment
    #plt.show()

    # for NoteBook

    from IPython.display import Image, display
    display(Image("plot.png"))


if __name__ == "__main__":
    main()