# MediCheck-GPT

Small character-level GPTs, built from scratch in PyTorch and trained on CMS Medicare inpatient hospital records. They do three jobs: explain a billing record in plain English (learned from an LLM teacher), write records in the CMS format, and flag records that are well-formed but wrong.

Built for the Open Avenues Build Fellowship project *Build, Train, and Deploy a Transformer Language Model in PyTorch* (Kacper Raczy), Fall 2026.

## Why this exists

My first prototype, trained on Michigan and Ohio hospitals, wrote records that read well and were wrong:

> Roled Hospital in Trenton, OH billed DRG 880 (SEPTICEMIA OR SEVERE SEPSIS WITHOUT MV >96 HOURS WITHOUT MCC) for 26 discharges: covered charges $18740, total payment $34406, Medicare payment $41082.

DRG 880 in the data is *Acute Adjustment Reaction and Psychosocial Dysfunction*, not sepsis, and the record says Medicare paid more than the total payment. In four of the seven records that run generated, Medicare paid more than the total. A schema check would pass it: no nulls, the right types, every field filled. That is a **silent error**, and real claims pipelines carry the same kind. I turned the model around: if a model trained on clean records finds a record surprising, the record deserves a second look.

The same records are unreadable to most people who pay for them. Following my mentor Kacper Raczy's suggestion, an LLM writes a one-sentence plain-English explanation of each record, and a GPT more than 1,000 times smaller learns to write those explanations itself. Because an LLM writing about numbers is its own source of silent errors, Python computes every number the LLM may use, and any sentence containing a number that is not in the record is thrown out.

## The flow

| Stage | What happens | Code |
|---|---|---|
| 1. Data | CMS CSV → one English sentence per row → split **by hospital** (160 / 20 / 20) so test hospitals are never seen in training | `src/data.py` |
| 2. Tokenize | Character level, vocabulary of 77. Each digit is its own token, which matters when the job is spotting a wrong number | `src/data.py` |
| 2b. Teacher | An LLM (Qwen2.5-1.5B-Instruct, on the T4) writes a plain-English sentence per record from facts Python computed; sentences with invented or mislabeled numbers are rejected | `src/teacher.py` |
| 3. Baseline | Bigram model: one character of memory | `src/model.py` |
| 4. Transformer | Decoder-only GPT: 4 layers, 4 heads, 128-dim embeddings, context 271 (one full record) | `src/model.py` |
| 5. Train | AdamW, cosine learning rate with warmup, weight decay, early stopping, best checkpoint kept; perplexity, ROUGE-L and record validity reported at every evaluation | `src/train.py` |
| 6. Generate | Temperature and top-k sampling; stops when the model ends the record | `src/generate.py` |
| 7. Evaluate | Perplexity, token accuracy, ROUGE-L, BLEU, BERTScore against the teacher's sentences; record validity; whether the student's numbers come from the record | `src/metrics.py` |
| 8. Flag | Plant four kinds of silent error in unseen hospitals' records; score surprise after the hospital name, relative to the bigram; ROC AUC with 95% intervals | `src/flag.py` |

## Data

CMS, *Medicare Inpatient Hospitals – by Provider and Service*, data year 2024 ([data.cms.gov](https://data.cms.gov)). One row is one hospital × one diagnosis-related group (DRG). The national file has 145,879 rows; I keep Michigan and Ohio: 9,640 rows across 200 hospitals.

Each row becomes one line:

```
Trinity Health Livonia Hospital in Livonia, MI billed DRG 057 (DEGENERATIVE NERVOUS SYSTEM DISORDERS WITHOUT MCC) for 21 discharges: covered charges $31729, total payment $13056, Medicare payment $7768.
```

The data is public and aggregated to hospital × DRG averages, so it carries no patient-level information. It is not committed here; `--prepare` builds everything from the CMS file.

## Usage

```bash
pip install -r requirements.txt

# 1. data
./gpt.py --input MUP_INP_RY26_P03_V10_DY24_PrvSvc.CSV --prepare data/processed

# 2. train: baseline, then transformer
./gpt.py --input data/processed --train checkpoints/bigram.pth --model bigram --context_size 8 --lr 1e-2 --steps 3000
./gpt.py --input data/processed --train checkpoints/gpt.pth --steps 10000

# 2b. teacher writes descriptions, student learns them
./gpt.py --input data/processed --teacher data/described --limit 5000
./gpt.py --input data/described --train checkpoints/explain.pth --init_from checkpoints/gpt.pth --steps 4000
./gpt.py --eval checkpoints/explain.pth --interactive            # paste a record, get a sentence

# 3. generate
./gpt.py --eval checkpoints/gpt.pth --prompt "Trinity Health Livonia Hospital in Livonia, MI billed DRG "
./gpt.py --eval checkpoints/gpt.pth --interactive

# 4. evaluate and flag
./gpt.py --input data/processed --evaluate checkpoints/gpt.pth --hf_metrics
./gpt.py --input data/processed --flag checkpoints/gpt.pth --background checkpoints/bigram.pth
./gpt.py --flag checkpoints/gpt.pth --flag_file my_records.txt     # rank any file of records, most suspicious first
```

Every hyperparameter is a flag (`--init_from`, `--n_layer`, `--n_head`, `--n_embd`, `--context_size`, `--dropout`, `--batch_size`, `--lr`, `--warmup`, `--weight_decay`, `--steps`, `--patience`, `--temperature`, `--top_k`). `./gpt.py -h` lists them. `notebooks/run_on_colab.ipynb` runs the whole flow on a free Colab T4. `WALKTHROUGH.md` explains the code step by step, in reading order.

## Results

All numbers are from run 1, saved with its outputs as `notebooks/run1_14nov.ipynb` (Colab T4, 14 November 2026). Test set = 20 hospitals never seen in training.

### Language model: bigram vs GPT

| Metric | Bigram | GPT |
|---|---|---|
| Parameters | 5,929 | 846,285 |
| Best validation loss | 2.152 | 0.652 (step 4,000; early stop at 6,000) |
| Test perplexity | 8.64 | 1.94 |
| Test next-character accuracy | 32.9% | 85.0% |
| ROUGE-L, completing real records | 0.004 | 0.723 |
| BLEU | not run | 0.546 |
| BERTScore F1 (not rescaled) | not run | 0.963 |

The completion prompt includes the DRG code, so part of the ROUGE score is the model recalling the description for a code it was given. Raw BERTScore crowds near 1.0 for any two sentences in the same format; later runs report it rescaled against a baseline.

**Fluent is not the same as correct.** Rules that hold for real records, checked on 100 records the GPT wrote from scratch:

| Rule | Real test records | GPT-generated |
|---|---|---|
| Parses as a well-formed record | 100% | 91% |
| DRG code matches its description | 99.7% | **26%** |
| Medicare payment ≤ total payment | 100% | 71% |
| Total payment ≤ covered charges | 100% | 80% |
| Known hospital placed in its real city | n/a (test hospitals are unseen) | 96.8% |

Nine in ten generated records look right, and three in four pair a diagnosis code with the wrong diagnosis. That gap is the silent-error problem this project is about. (The 0.3% of real records that fail the DRG rule carry codes absent from the training hospitals.)

### Model size (same data, seed and early stopping)

| Model | Layers / heads / dim | Dropout | Parameters | Best step | Best validation loss | Validation perplexity |
|---|---|---|---|---|---|---|
| Small | 2 / 2 / 64 | 0.1 | 126,989 | 5,000 | 0.716 | 2.05 |
| Base | 4 / 4 / 128 | 0.1 | 846,285 | 4,000 | 0.652 | 1.92 |
| Large | 6 / 6 / 192 | 0.2 | 2,747,789 | 1,500 | 0.621 | 1.86 |

Bigger generalizes better here, and overfits sooner: the large model peaked at step 1,500 even with twice the dropout. A notebook prototype with no schedule and no stopping rule showed the same pattern, which is why training uses early stopping, weight decay and a cosine schedule.

### Explaining records: what run 1 taught

**The teacher.** Qwen2.5-1.5B-Instruct wrote a sentence for 4,749 records. 15% contained a number that was not in the record and were rejected (2,582 / 814 / 639 kept for train / val / test). The accepted sentences had a subtler fault: real numbers with the wrong label ("received $15,466 from Medicare" when $15,466 was the total; "95% of the charges" when 95% was Medicare's share of the payment). A number check catches invented numbers, not mislabeled ones.

**The student**, trained from scratch on random windows: test perplexity 2.51, ROUGE-L 0.312, BLEU 0.107, BERTScore 0.891 (not rescaled), and **0% of its sentences used the record's numbers**. Given Munson Healthcare Grayling Hospital, it wrote about a hospital in Detroit. It learned how the teacher writes, not to read what it is given.

**Run 2 (in progress)** fixes both. The teacher gets each fact as a labelled phrase and one example; a second check rejects any number sitting next to the wrong label, and a rejected record gets one retry. The student starts from the record GPT (pre-training) and is fine-tuned on whole examples with the loss on the explanation only (supervised fine-tuning, Lecture 6–7).

| Metric | Run 1 | Run 2 |
|---|---|---|
| Teacher sentences rejected | 15% (invented numbers only) | |
| Student ROUGE-L | 0.312 | |
| Student BLEU | 0.107 | |
| Student BERTScore F1 | 0.891 (raw) | (rescaled) |
| Student sentences using only the record's numbers | 0% | |

### Flagging silent errors: what run 1 taught

ROC AUC on 500 clean test records against the same records with one planted error. 0.5 = chance; below 0.5 = backwards.

| Planted error | Bigram | GPT, mean surprise | GPT, peak surprise |
|---|---|---|---|
| DRG code with another code's description | 0.610 | **0.680** | 0.513 |
| Hospital moved to another city in its state | 0.469 | **0.226** | 0.288 |
| Total and Medicare payments swapped | 0.500 | 0.526 | 0.500 |
| Covered charges with an extra digit | 0.509 | 0.553 | 0.500 |

The GPT catches mismatched diagnoses best, as expected for a model of text. Two results are worth more than a good score:

- **A wrong city scored less suspicious than the right one (0.226).** Planted cities come from training data, so they are familiar (Columbus); true cities of unseen hospitals are often rare (Grayling). Raw surprise measures familiarity, not correctness, the same failure Nalisnick et al. (2019) found in deep generative models.
- **Peak surprise tied at exactly 0.5** because every record's most surprising stretch is the name of a hospital the model has never seen, so the planted error never became the peak.

**Run 2 (in progress)** scores only what follows the hospital name, and divides out plain rarity by subtracting the bigram's surprise (a likelihood ratio, after Ren et al., 2019), with 95% bootstrap intervals. Arithmetic errors (swapped payments) remain a job for a one-line rule, and the project says so.

## Design decisions

- **Split by hospital, not by row.** A row split would put the same hospital's name and address in train and test, and the test score would measure memory, not generalization.
- **Characters, not BPE.** Records are short, and a subword tokenizer can merge `$52` into one token and `24` into another, which hides exactly the digit-level errors this project looks for.
- **Context 271.** The longest record is 269 characters; with its two newlines the model always sees a whole record at once.
- **The teacher phrases, Python counts.** Small LLMs misstate arithmetic. Every percentage is computed before the LLM sees the record, and every number it writes is checked against the record.
- **Surprise is scored after the hospital name, against a background.** An unseen name is new, not wrong; and a rare true value is not an error. Subtracting the bigram's surprise leaves what the GPT knows about context.
- **A language model is not a rule engine.** Arithmetic relations such as Medicare ≤ total are cheap to check with a rule. The model's value is the errors no one wrote a rule for. The detection table shows which error types it catches and which still need a rule.

## Limitations

Two states, one year, and aggregated data. A planted error is a stand-in for a real one. A record from a hospital type the model has never seen can look surprising without being wrong, so the flagger ranks records for review; it does not decide.

## Credits

- Model, bigram baseline and training loop adapted from Kacper Raczy's course notebooks (`baseline.ipynb`, `transformer.ipynb`), which follow Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) lectures. Changes are marked in `src/model.py`.
- Teacher model: Qwen2.5-1.5B-Instruct (Qwen team, Alibaba), via Hugging Face `transformers`. Teacher–student training follows Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network*, 2015; the LLM-reference approach was Kacper Raczy's recommendation for this dataset.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Georgia Tech Polo Club), shared in the course, for the attention walkthrough.
- Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019; Ren et al., *Likelihood Ratios for Out-of-Distribution Detection*, NeurIPS 2019.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Evaluation metrics as taught in the course (Lecture 6–7); BLEU and BERTScore through Hugging Face [`evaluate`](https://github.com/huggingface/evaluate).
- Data: Centers for Medicare & Medicaid Services.

Cynthia Mutua · [cynthiamutua.com](https://cynthiamutua.com)
