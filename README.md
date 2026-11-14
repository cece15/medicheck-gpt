# MediCheck-GPT

Small character-level GPTs, built from scratch in PyTorch and trained on CMS Medicare inpatient hospital records. They do three jobs: explain a billing record in plain English (learned from an LLM teacher), write records in the CMS format, and flag records that are well-formed but wrong.

Built for the Open Avenues Build Fellowship project *Build, Train, and Deploy a Transformer Language Model in PyTorch* (Kacper Raczy), Fall 2026.

## Why this exists

My first model, trained on Michigan and Ohio hospitals, wrote records that read well and were wrong:

> University Of Michigan Health in Ann Arbor, MI billed DRG 556 (PATHOLOWERY AND ND URINARY TRACT DIAGNOSES WITH MCC) for 23 discharges: covered charges $35109, total payment $12256, Medicare payment $13593.

DRG 556 in the data is *Signs and Symptoms of Musculoskeletal System and Connective Tissue without MCC*, and the record says Medicare paid more than the total payment. In all six records that run generated, the Medicare payment was larger than the total payment. A schema check would pass it: no nulls, the right types, every field filled. That is a **silent error**, and real claims pipelines carry the same kind. I turned the model around: if a model trained on clean records finds a record surprising, the record deserves a second look.

The same records are unreadable to most people who pay for them. Following my mentor Kacper Raczy's suggestion, an LLM writes a one-sentence plain-English explanation of each record, and a GPT more than 1,000 times smaller learns to write those explanations itself. Because an LLM writing about numbers is its own source of silent errors, Python computes every number the LLM may use, and any sentence containing a number that is not in the record is thrown out.

## The flow

| Stage | What happens | Code |
|---|---|---|
| 1. Data | CMS CSV → one English sentence per row → split **by hospital** (160 / 20 / 20) so test hospitals are never seen in training | `src/data.py` |
| 2. Tokenize | Character level, vocabulary of 77. Each digit is its own token, which matters when the job is spotting a wrong number | `src/data.py` |
| 2b. Teacher | An LLM (Qwen2.5-1.5B-Instruct, on the T4) writes a plain-English sentence per record from facts Python computed; sentences with invented numbers are rejected | `src/teacher.py` |
| 3. Baseline | Bigram model: one character of memory | `src/model.py` |
| 4. Transformer | Decoder-only GPT: 4 layers, 4 heads, 128-dim embeddings, context 271 (one full record) | `src/model.py` |
| 5. Train | AdamW, cosine learning rate with warmup, weight decay, early stopping, best checkpoint kept; perplexity, ROUGE-L and record validity reported at every evaluation | `src/train.py` |
| 6. Generate | Temperature and top-k sampling; stops when the model ends the record | `src/generate.py` |
| 7. Evaluate | Perplexity, token accuracy, ROUGE-L, BLEU, BERTScore against the teacher's sentences; record validity; whether the student's numbers come from the record | `src/metrics.py` |
| 8. Flag | Plant four kinds of silent error in unseen hospitals' records; measure whether the model ranks them above clean ones (ROC AUC) | `src/flag.py` |

## Data

CMS, *Medicare Inpatient Hospitals – by Provider and Service*, data year 2024 ([data.cms.gov](https://data.cms.gov)). One row is one hospital × one diagnosis-related group (DRG). The national file has 145,879 rows; I keep Michigan and Ohio: 9,640 rows across 200 hospitals.

Each row becomes one line:

```
Trinity Health Livonia Hospital in Livonia, MI billed DRG 057 (DEGENERATIVE NERVOUS SYSTEM DISORDERS WITHOUT MCC) for 21 discharges: covered charges $31729, total payment $13056, Medicare payment $7768.
```

The data is public and aggregated (CMS suppresses any hospital–DRG cell with fewer than 11 discharges), so it carries no patient-level information. It is not committed here; `--prepare` builds everything from the CMS file.

## Usage

```bash
pip install -r requirements.txt

# 1. data
./gpt.py --input MUP_INP_RY26_P03_V10_DY24_PrvSvc.CSV --prepare data/processed

# 2. train: baseline, then transformer
./gpt.py --input data/processed --train checkpoints/bigram.pth --model bigram --context_size 8 --lr 1e-2 --steps 3000
./gpt.py --input data/processed --train checkpoints/gpt.pth --steps 10000

# 2b. teacher writes descriptions, student learns them
./gpt.py --input data/processed --teacher data/described --limit 3000
./gpt.py --input data/described --train checkpoints/explain.pth --steps 10000
./gpt.py --eval checkpoints/explain.pth --interactive            # paste a record, get a sentence

# 3. generate
./gpt.py --eval checkpoints/gpt.pth --prompt "Trinity Health Livonia Hospital in Livonia, MI billed DRG "
./gpt.py --eval checkpoints/gpt.pth --interactive

# 4. evaluate and flag
./gpt.py --input data/processed --evaluate checkpoints/gpt.pth --hf_metrics
./gpt.py --input data/processed --flag checkpoints/gpt.pth
./gpt.py --flag checkpoints/gpt.pth --flag_file my_records.txt     # rank any file of records, most suspicious first
```

Every hyperparameter is a flag (`--n_layer`, `--n_head`, `--n_embd`, `--context_size`, `--dropout`, `--batch_size`, `--lr`, `--warmup`, `--weight_decay`, `--steps`, `--patience`, `--temperature`, `--top_k`). `./gpt.py -h` lists them. `notebooks/run_on_colab.ipynb` runs the whole flow on a free Colab T4. `WALKTHROUGH.md` explains the code step by step, in reading order.

## Results

**First run, 14 November (Colab T4, PyTorch 2.14, 3,000 steps, constant learning rate, no early stopping).**

| Model | Parameters | Best validation loss | Note |
|---|---|---|---|
| Bigram | 5,929 | 2.094 (step 2,999) | Generates character soup: no record structure. |
| GPT | 0.85M | 0.665 (step 1,500) | Validation rose to 0.685 by step 2,999 while training fell to 0.284: overfitting. This is why the final run uses early stopping, weight decay and a cosine schedule. |

Training took 165 seconds. Reloading the saved checkpoint reproduced the same generation exactly under a fixed seed.

**Final run.** *To be filled from `notebooks/run_on_colab.ipynb`.*

| Metric | Bigram | GPT |
|---|---|---|
| Test perplexity | | |
| Test next-character accuracy | | |
| ROUGE-L (completion of real records) | | |
| BERTScore F1 | | |
| Generated records that parse | | |
| Generated records with the right DRG description | | |
| Generated records with Medicare ≤ total payment | | |

**Hyperparameter comparison (same data, seed and early stopping).** *To be filled.*

| Model | Layers | Heads | Embedding | Dropout | Parameters | Best validation loss | Validation perplexity |
|---|---|---|---|---|---|---|---|
| Small | 2 | 2 | 64 | 0.1 | | | |
| Base | 4 | 4 | 128 | 0.1 | | | |
| Large | 6 | 6 | 192 | 0.2 | | | |

**Explaining records (student vs LLM teacher, unseen hospitals).** *To be filled.*

| Metric | Value |
|---|---|
| Teacher sentences rejected for invented numbers | |
| Student ROUGE-L | |
| Student BLEU | |
| Student BERTScore F1 | |
| Student sentences whose numbers all come from the record | |

**Silent-error detection (ROC AUC; 0.5 = chance).** *To be filled.*

| Planted error | Bigram | GPT |
|---|---|---|
| DRG code with the wrong description | | |
| Hospital moved to another city in its state | | |
| Total and Medicare payments swapped | | |
| Covered charges with an extra digit | | |

## Design decisions

- **Split by hospital, not by row.** A row split would put the same hospital's name and address in train and test, and the test score would measure memory, not generalization.
- **Characters, not BPE.** Records are short, and a subword tokenizer can merge `$52` into one token and `24` into another, which hides exactly the digit-level errors this project looks for.
- **Context 271.** The longest record is 269 characters; with its two newlines the model always sees a whole record at once.
- **The teacher phrases, Python counts.** Small LLMs misstate arithmetic. Every percentage is computed before the LLM sees the record, and every number it writes is checked against the record.
- **Two surprise scores.** The mean loss over a record catches a record that is odd throughout; the peak over any 8-character window catches one wrong field, which is what most silent errors look like.
- **A language model is not a rule engine.** Arithmetic relations such as Medicare ≤ total are cheap to check with a rule. The model's value is the errors no one wrote a rule for. The detection table shows which error types it catches and which still need a rule.

## Limitations

Two states, one year, and aggregated data. A planted error is a stand-in for a real one. A record from a hospital type the model has never seen can look surprising without being wrong, so the flagger ranks records for review; it does not decide.

## Credits

- Model, bigram baseline and training loop adapted from Kacper Raczy's course notebooks (`baseline.ipynb`, `transformer.ipynb`), which follow Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) lectures. Changes are marked in `src/model.py`.
- Teacher model: Qwen2.5-1.5B-Instruct (Qwen team, Alibaba), via Hugging Face `transformers`. Teacher–student training follows Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network*, 2015; the LLM-reference approach was Kacper Raczy's recommendation for this dataset.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Evaluation metrics as taught in the course (Lecture 6–7); BLEU and BERTScore through Hugging Face [`evaluate`](https://github.com/huggingface/evaluate).
- Data: Centers for Medicare & Medicaid Services.

Cynthia Mutua · [cynthiamutua.com](https://cynthiamutua.com)
