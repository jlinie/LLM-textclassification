# src/llm_evaluate.py
import argparse
import json
import time
from pathlib import Path
import pickle

import pandas as pd
from tqdm import tqdm
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report
import ollama
from concurrent.futures import ThreadPoolExecutor

def main():
    parser = argparse.ArgumentParser(
        description="LLM classification + TF–IDF filter + NER few-shot prompt"
    )
    parser.add_argument("--test",           type=Path, required=True)
    parser.add_argument("--model-name",     required=True)
    parser.add_argument("--tfidf-threshold",type=float, default=0.1)
    parser.add_argument("--out-preds",      type=Path,
                        default=Path("results/llm_preds.json"))
    parser.add_argument("--workers",        type=int, default=None,
                        help="# of threads for the LLM calls")
    args = parser.parse_args()

    # 1) load data
    df = pd.read_csv(args.test, sep="\t")
    df["Entities"] = df.get("Entities", "")  # fill missing
    texts         = df["Text"].tolist()
    entities_list = df["Entities"].tolist()
    y_true        = df["Category"].tolist()

    # 2) load or fit TF–IDF vectorizer (with caching)
    args.out_preds.parent.mkdir(parents=True, exist_ok=True)
    vect_cache = args.out_preds.parent / "vectorizer.pkl"
    if vect_cache.exists():
        with open(vect_cache, "rb") as f:
            vectorizer = pickle.load(f)
    else:
        vectorizer = TfidfVectorizer(max_features=10_000)
        vectorizer.fit(texts)
        with open(vect_cache, "wb") as f:
            pickle.dump(vectorizer, f)

    # precompute which examples get the LLM vs. fallback
    X = vectorizer.transform(texts)
    max_scores = X.max(axis=1).toarray().ravel()        
    needs_llm  = max_scores >= args.tfidf_threshold
    fallback   = df["Category"].mode()[0]
    categories = sorted(df["Category"].unique())

    # 3) few‐shot prompt prefix
    categories_str = ", ".join(categories)
    few_shot = (
        "Sen bir metin sınıflandırma asistanısın. Aşağıda iki örnek ve bir "
        "sınıflandırma işi var. Tüm **Entities** bilgilerini kullanarak, her "
        "cümleyi yalnızca aşağıdaki kategorilerden birine atayın. Cevap olarak "
        "**sadece** kategori adını (tek kelime) yazın; başka metin veya "
        "noktalama eklemeyin.\n\n"
        f"Kategoriler: {categories_str}\n\n"
        "Örnek 1:\n"
        "Cümle: Walt Disney Music Company tarafından Musicnotes.com'da yayımlanan nota kağıdına göre şarkı , üç dörtlük ölçü işaretiyle ve dakikada 192 tempoluk melodiyle yazılmıştır .\n"
        "Entities: B-artist_name=Walt Disney Music Company, "
        "B-composition_form=üç dörtlük ölçü işareti\n"
        "Kategori: music\n\n"
    )

    # 4) helper to call the LLM
    def call_llm(text, entities):
        # build entities block
        ent_lines = []
        for item in entities.split(","):
            if "=" in item:
                et, val = item.split("=", 1)
                ent_lines.append(f"{et.strip()}: {val.strip()}")
        ent_block = "\n".join(ent_lines)

        # assemble prompt
        p = few_shot + f"Cümle: {text}\n"
        if ent_block:
            p += "Entities:\n" + ent_block + "\n"
        p += "Yanıt:"

        # invoke
        resp = ollama.chat(
            model=args.model_name,
            messages=[{"role": "user", "content": p}],
            options={"max_tokens": 1, "temperature": 0.0, "stop": ["\n"]},
        )

        # extract single-word
        if isinstance(resp, dict):
            if "choices" in resp:
                content = resp["choices"][0]["message"]["content"]
            elif "message" in resp:
                content = resp["message"]["content"]
            else:
                content = str(resp)
        else:
            content = str(resp)

        pred = content.strip()
        return pred if pred in categories else fallback

    # 5) seed preds with fallback
    n = len(texts)
    y_pred = [fallback] * n
    to_call = [i for i, flag in enumerate(needs_llm) if flag]

    # 6) run LLM calls with a timer + progress bar
    start_all = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as exe:
        for idx, pred in tqdm(
            zip(to_call, exe.map(
                call_llm,
                (texts[i] for i in to_call),
                (entities_list[i] for i in to_call),
            )),
            total=len(to_call),
            desc="LLM classify"
        ):
            y_pred[idx] = pred
    total_time = time.time() - start_all

    # 7) save JSON with timings
    args.out_preds.parent.mkdir(exist_ok=True, parents=True)
    with open(args.out_preds, "w", encoding="utf-8") as f:
        json.dump({
            "y_true": y_true,
            "y_pred": y_pred,
            "llm_total_time_s": total_time,
            "llm_avg_time_per_example_s": total_time / len(to_call) if to_call else 0.0
        }, f, ensure_ascii=False, indent=2)

    # 8) print final metrics + runtime
    print(f"LLM accuracy: {accuracy_score(y_true, y_pred):.4f}")
    print(f"LLM total time: {total_time:.1f}s  (avg {total_time/len(to_call):.2f}s/LLM-call)")
    print(classification_report(y_true, y_pred, zero_division=0))

if __name__ == "__main__":
    main()
