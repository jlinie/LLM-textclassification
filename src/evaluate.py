import argparse
from pathlib import Path
import joblib
from src.utils import load_split
from sklearn.metrics import classification_report, accuracy_score

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--test",  required=True, type=Path)
    args = p.parse_args()

    data = joblib.load(args.model)
    vectorizer, clf = data["vectorizer"], data["model"]

    X_test, y_test = load_split(args.test)
    Xt = vectorizer.transform(X_test)
    preds = clf.predict(Xt)

    print("Test accuracy:", accuracy_score(y_test, preds))
    print(classification_report(y_test, preds, zero_division=0))

if __name__ == "__main__":
    main()
