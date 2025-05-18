from sklearn.feature_extraction.text import TfidfVectorizer

def build_vectorizer(max_features=50000, ngram_range=(1,2), min_df=5):
    return TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        min_df=min_df,
        strip_accents="unicode",
        lowercase=True
    )
