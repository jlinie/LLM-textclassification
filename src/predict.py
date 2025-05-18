import argparse
from pathlib import Path
import joblib

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, type=Path)
    p.add_argument("--text",  required=True)
    args = p.parse_args()

    data = joblib.load(args.model)
    vec, clf = data["vectorizer"], data["model"]
    x = vec.transform([args.text])
    print("Predicted category:", clf.predict(x)[0])

if __name__ == "__main__":
    main()
