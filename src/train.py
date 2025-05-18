import argparse
from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from src.utils import load_split
from src.features import build_vectorizer

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, type=Path)
    p.add_argument("--val",   required=True, type=Path)
    p.add_argument("--out",   required=True, type=Path, help="models/logreg.pkl")
    args = p.parse_args()

    X_train, y_train = load_split(args.train)
    X_val,   y_val   = load_split(args.val)

    vectorizer = build_vectorizer()
    Xtr = vectorizer.fit_transform(X_train)

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="saga",
        C=1.0,
        max_iter=1000,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(Xtr, y_train)

    # validation
    Xv = vectorizer.transform(X_val)
    val_acc = clf.score(Xv, y_val)
    print(f"Validation accuracy: {val_acc:.4f}")

    # kaydet
    args.out.parent.mkdir(exist_ok=True, parents=True)
    joblib.dump({"vectorizer": vectorizer, "model": clf}, args.out)

if __name__ == "__main__":
    main()
