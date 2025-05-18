import argparse
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split


def split_data(
    input_tsv: Path,
    output_dir: Path,
    val_frac: float,
    test_frac: float,
    seed: int
):
    # 1) read raw TSV with columns: Category, NER (to be treated as Entities), Text
    df = pd.read_csv(input_tsv, sep='\t', header=None)
    df.columns = ['Category', 'Entities', 'Text']

    # 2) split off (val + test)
    train, temp = train_test_split(
        df,
        test_size=val_frac + test_frac,
        random_state=seed,
        stratify=df['Category']
    )

    # 3) split temp into validation and test
    rel_test = test_frac / (val_frac + test_frac)
    validation, test = train_test_split(
        temp,
        test_size=rel_test,
        random_state=seed,
        stratify=temp['Category']
    )

    # 4) write out splits, retaining Text, Category, and Entities
    output_dir.mkdir(parents=True, exist_ok=True)
    for split_df, name in [
        (train, 'train.tsv'),
        (validation, 'validation.tsv'),
        (test, 'test.tsv')
    ]:
        split_df[['Text', 'Category', 'Entities']].to_csv(
            output_dir / name,
            sep='\t',
            index=False
        )


def main():
    parser = argparse.ArgumentParser(
        description="Split TWNERTC data into train/val/test with Entities column"
    )
    parser.add_argument(
        "-i", "--input",
        type=Path,
        required=True,
        help="Path to the TWNERTC .tsv file (with no header, columns: Category, NER, Text)"
    )
    parser.add_argument(
        "-o", "--output-dir",
        type=Path,
        required=True,
        help="Directory where train/validation/test.tsv will be written"
    )
    parser.add_argument(
        "--val-frac",
        type=float,
        default=0.1,
        help="Fraction of data for validation (default: 0.1)"
    )
    parser.add_argument(
        "--test-frac",
        type=float,
        default=0.1,
        help="Fraction of data for final test (default: 0.1)"
    )
    parser.add_argument(
        "--seed", "-s",
        type=int,
        default=42,
        help="Random seed for reproducible splits"
    )
    args = parser.parse_args()

    split_data(
        input_tsv=args.input,
        output_dir=args.output_dir,
        val_frac=args.val_frac,
        test_frac=args.test_frac,
        seed=args.seed
    )


if __name__ == "__main__":
    main()
