RAW_DATA      := data/TWNERTC_TC_Fine\ Grained\ NER_DomainIndependent_NoiseReduction.tsv
PROCESSED_DIR := data/processed
MODEL         := models/logreg_baseline.pkl
RESULTS_DIR   := results

VAL_FRAC      := 0.1
TEST_FRAC     := 0.1
SEED          := 42
LLAMA_MODEL   := llama3.2
TFIDF_THRESH  ?= 0.1

.PHONY: all split train eval predict llm clean

all: split train eval llm

split:
	python3 -m data_manipulation.split \
		--input $(RAW_DATA) \
		--output-dir $(PROCESSED_DIR) \
		--val-frac $(VAL_FRAC) \
		--test-frac $(TEST_FRAC)

train:
	python3 -m src.train \
		--train $(PROCESSED_DIR)/train.tsv \
		--val $(PROCESSED_DIR)/validation.tsv \
		--out $(MODEL)
		--out-report $(RESULTS_DIR)/logreg_report.json

eval:
	python3 -m src.evaluate \
		--model $(MODEL) \
		--test $(PROCESSED_DIR)/test.tsv

predict:
	python3 -m src.predict \
		--model $(MODEL) \
		--text "Bu cümle hangi kategoride?"

llm:
	mkdir -p $(RESULTS_DIR)
	python3 src/llm_evaluate.py \
		--test $(PROCESSED_DIR)/test.tsv \
		--model-name $(LLAMA_MODEL) \
		--tfidf-threshold $(TFIDF_THRESH) \
		--out-preds $(RESULTS_DIR)/llm_preds_$(LLAMA_MODEL).json

clean:
	rm -rf $(PROCESSED_DIR) $(MODEL) $(RESULTS_DIR)
