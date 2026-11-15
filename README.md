# MediCheck-GPT

Small character-level GPTs, built from scratch in PyTorch and trained on CMS Medicare inpatient hospital records. They do three jobs: explain a billing record in plain English (learned from an LLM teacher), write records in the CMS format, and, with simple rules, flag records that are well-formed but wrong.

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
| 8. Flag | Plant four kinds of silent error in unseen hospitals' records; measure ROC AUC with 95% intervals and a paired test; for real files, rules first, then the model's surprise | `src/flag.py` |

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
./gpt.py --input data/described --train checkpoints/explain.pth --init_from checkpoints/gpt.pth --delex --steps 4000
./gpt.py --eval checkpoints/explain.pth --interactive --temperature 0.5
./gpt.py --eval checkpoints/explain.pth --explain_file records.txt --n 100   # explain a file, check every sentence            # paste a record, get a sentence

# 3. generate
./gpt.py --eval checkpoints/gpt.pth --prompt "Trinity Health Livonia Hospital in Livonia, MI billed DRG "
./gpt.py --eval checkpoints/gpt.pth --interactive

# 4. evaluate and flag
./gpt.py --input data/processed --evaluate checkpoints/gpt.pth --hf_metrics
./gpt.py --input data/processed --flag checkpoints/gpt.pth --background checkpoints/bigram.pth
./gpt.py --input data/processed --flag checkpoints/gpt.pth --flag_file my_records.txt   # rules first, then the model's ranking
```

Every hyperparameter is a flag (`--init_from`, `--delex`, `--n_layer`, `--n_head`, `--n_embd`, `--context_size`, `--dropout`, `--batch_size`, `--lr`, `--warmup`, `--weight_decay`, `--steps`, `--patience`, `--temperature`, `--top_k`). `./gpt.py -h` lists them. `notebooks/run_on_colab.ipynb` runs the whole flow on a free Colab T4. `WALKTHROUGH.md` explains the code step by step, in reading order.

## Results

Runs on a Colab T4, 14 November 2026, each saved with its outputs: `notebooks/run1_14nov.ipynb` (1a), `notebooks/run1b_14nov.ipynb` (1b, a repeat of 1a), `notebooks/run2_14nov.ipynb` (2), `notebooks/run3a_14nov.ipynb` (3a: re-evaluation and the paired test) and `notebooks/run3_14nov.ipynb` (3: the placeholder explainer). The base GPT was trained three times with the same settings and seed; ranges below are across those trainings (GPU arithmetic is not perfectly deterministic). The bigram matched exactly every time. Test set: 20 hospitals never seen in training.

### Language model: bigram vs GPT

| Metric | Bigram | GPT (range over 3 trainings) |
|---|---|---|
| Parameters | 5,929 | 846,285 |
| Best validation loss | 2.152 | 0.635 – 0.652 |
| Test perplexity | 8.64 | 1.94 – 1.96 |
| Test next-character accuracy | 32.9% | 84.8% – 85.0% |
| ROUGE-L, completing real records | 0.002 – 0.004 | 0.713 – 0.725 |
| BLEU | not run | 0.537 – 0.551 |
| BERTScore F1 | not run | 0.961 – 0.963 raw; 0.796 rescaled (run 2) |

The completion prompt includes the DRG code, so part of the ROUGE score is the model recalling the description for a code it was given. Raw BERTScore crowds near 1.0 for any two sentences in the same format, which is why run 2 rescales it.

**Fluent is not the same as correct.** Rules that hold for real records, checked on 100 records the GPT wrote from scratch:

| Rule | Real test records | GPT-generated (range over 3 runs) |
|---|---|---|
| Parses as a well-formed record | 100% | 88% – 96% |
| DRG code matches its description | 99.7% | **26% – 40%** |
| Medicare payment ≤ total payment | 100% | 63% – 80% |
| Total payment ≤ covered charges | 100% | 80% – 85% |
| Known hospital placed in its real city | n/a (test hospitals are unseen) | 92% – 97% |

Nearly every generated record looks right, and most pair a diagnosis code with the wrong diagnosis. That gap is the silent-error problem this project is about. (The 0.3% of real records that fail the DRG rule carry codes absent from the training hospitals.)

### Model size (same data, seed and early stopping)

| Model | Layers / heads / dim | Dropout | Parameters | Best validation loss, each training |
|---|---|---|---|---|
| Small | 2 / 2 / 64 | 0.1 | 126,989 | 0.716, 0.718, 0.743 |
| Base | 4 / 4 / 128 | 0.1 | 846,285 | 0.652, 0.635, 0.640 |
| Large | 6 / 6 / 192 | 0.2 | 2,747,789 | 0.621, 0.634 |

Small is clearly worse. Base and large overlap across trainings, so three times the parameters bought no reliable gain on 1.6M characters, and the large model overfit by step 1,500 even with twice the dropout. Base is the right size for this much data. One training would have suggested otherwise; repeating it is how this was caught.

### Explaining records: three attempts

**The teacher (Qwen2.5-1.5B-Instruct).**

| | Run 1 | Run 2 |
|---|---|---|
| How facts were given | labelled values | full phrases plus one example sentence |
| Checks | invented numbers | invented numbers and numbers next to the wrong label |
| First-attempt sentences with an invented number | 15% | 19% |
| First-attempt sentences with a mislabeled number | not checked | 44% |
| Records kept (after one retry in run 2) | 85% | 66% (3,328 train / 594 val / 518 test) |

Run 1's accepted sentences were fluent and often wrong ("received $15,466 from Medicare" when $15,466 was the total). Run 2's check showed how often: on a first attempt, a 1.5B model attached a real number to the wrong label in 44% of sentences. Every sentence used for training now passes both checks. The cost is a third of the records, and the kept set may lean toward records that are easier to describe.

**The student (0.89M parameters).**

| | Run 1 | Run 2 | Run 3 |
|---|---|---|---|
| Training | from scratch, random windows | from the record GPT, whole examples, loss on the explanation only | as run 2, with the record's values as placeholders |
| ROUGE-L vs teacher | 0.312 / 0.319 | 0.438 / 0.413 | **0.561** |
| BLEU | 0.107 / 0.105 | 0.217 / 0.192 | **0.459** |
| BERTScore F1 | 0.891 / 0.892 raw | 0.513 / 0.504 rescaled | **0.676** rescaled |
| Explanation-only perplexity / next-character accuracy | | 1.88 / 85.7% | **1.15 / 95.9%** |
| Sentences using only the record's numbers | 0% | 0% | **93%** |
| Sentences passing the teacher's own checks | | 0% | **93%** |
| Sentences naming the record's hospital or city | | 2% | **100%** |

Run 1 learned the teacher's style and ignored the record. Run 2 writes like the teacher (explanation perplexity 1.88, 86% of next characters right) and still fails every factual check: none of its sentences pass the teacher's checks, and 2% name the right hospital or city. It reads the record only partly: for Munson Healthcare Grayling Hospital it wrote about Munson Healthcare Otsego Memorial in Gaylord, the same health system thirty miles away, with numbers of the right size that were not the record's. A model under a million parameters fills a sentence from memory rather than copying digits from its input; the explainer reproduced the very silent error the project studies. Run 3 stops asking it to copy: the teacher's numbers, hospital, city and condition become placeholders for training, the student learns where each goes, and code puts the record's values back. This is delexicalisation (Wen et al., 2015), the standard way data-to-text systems keep numbers faithful.

It worked. For Munson Healthcare Grayling Hospital, run 3 wrote: *"Munson Healthcare Grayling Hospital in Grayling, MI, treated 27 Medicare patients with respiratory infections and inflammations, averaging a bill of $37,016 but receiving $15,466 in total payments, with Medicare covering 95% of those payments."* Every number is the record's. Across 100 unseen test records, 93% of sentences pass the same checks the teacher had to pass, against 0% in run 2.

What is still free text is the condition when the teacher paraphrased it: in one sample the model called respiratory infections "chronic obstructive pulmonary disease". The remaining 7% of failures are numbers that slipped in outside a placeholder. Both are visible to the same checks, so in use every sentence would be checked before it is shown.

(Run 2's evaluation reported a test perplexity of 33.8. That was a measurement error: it scored the record text, which the fine-tuned student is never trained to predict. Run 3 measures perplexity on the explanation only.)

### Flagging silent errors

Two questions, two tests. **Ranking (ROC AUC):** across different records, do corrupted ones score higher than clean ones? **Noticing (paired):** on the *same* record, does the planted error raise the model's surprise? 500 clean test records; 0.5 = chance, below 0.5 = backwards.

| Planted error | Bigram, raw AUC | GPT raw AUC (runs 1a / 1b / 2) | GPT raw AUC 95% CI | **GPT paired** (base / large) |
|---|---|---|---|---|
| DRG code with another code's description | 0.610 | 0.680 / 0.682 / 0.677 | 0.645 – 0.703 | **0.848 / 0.846** |
| Hospital moved to another city in its state | 0.469 | 0.226 / 0.210 / 0.208 | 0.183 – 0.233 | 0.164 / 0.202 |
| Total and Medicare payments swapped | 0.500 | 0.526 / 0.528 / 0.529 | 0.495 – 0.556 | **0.896 / 0.766** |
| Covered charges with an extra digit | 0.509 | 0.553 / 0.545 / 0.542 | 0.507 – 0.571 | **0.994 / 1.000** |

Run 2 also tried scoring only after the hospital name, and a likelihood ratio against the bigram (Ren et al., 2019). Both lowered diagnosis detection (0.68 → 0.52) and are kept in `flag.py` as documented negative results; the raw score stays.

What this shows:

- **The model notices almost every error; it cannot rank them.** On the same record, an extra digit raises surprise 99% of the time, swapped payments 90%, a mismatched diagnosis 85%. Across records the same errors sit near chance (0.53–0.68), because one hospital's ordinary record can be more surprising than another's corrupted one. The signal is real; the noise is the hospitals. Earlier runs, which measured only ranking, concluded the model "could not see arithmetic". The paired test shows that was wrong: it sees it, it just cannot tell you which record to look at first.
- **Surprise measures familiarity, not truth.** The wrong city is the one error the model gets backwards in both tests (0.16–0.21): a planted city comes from training data (Columbus) and is familiar; the true city of an unseen hospital is often rare (Grayling). The same failure is documented for deep generative models (Nalisnick et al., 2019). A wrong city is a lookup against CMS's hospital register, not a language problem.
- **So the tool is a hybrid.** `--flag_file` runs rules first (Medicare more than the total; paid more than billed; a bill beyond the 99.5th percentile of charge-to-payment ratios in training), then ranks the rest by the model's surprise. On the demo file (5 real, 5 planted, shuffled), the rules put the swapped payments and the extra digit at the top, and the model ranked two of the remaining three planted records 2nd and 3rd of 8: four of the top five are planted errors.

## Testing

```bash
python -m pytest tests -q        # 13 tests, about a minute on a laptop CPU, no data or GPU needed
```

- `tests/test_core.py` covers the parts that must never be wrong: parsing, the hospital split (no hospital in two sets), the tokenizer, that fine-tuning grades only the explanation, the placeholder round trip, every kind of mistake the teacher's checks must catch (and correct sentences they must pass), the AUC against its brute-force definition, the rules, that each planted error changes exactly one field, that the GPT learns, and that weights transfer by character.
- `tests/test_end_to_end.py` runs the whole command-line flow on a small synthetic file, from CSV to a flagged record, and checks that a planted arithmetic error is caught by the rules and ranked first.
- Section 8 of the notebook tests generalization on Illinois, a state none of the models trained on: record-model perplexity, the explainer's check pass rate on 100 Illinois records (`--explain_file`), and flagging.

## Design decisions

- **Split by hospital, not by row.** A row split would put the same hospital's name and address in train and test, and the test score would measure memory, not generalization.
- **Characters, not BPE.** Records are short, and a subword tokenizer can merge `$52` into one token and `24` into another, which hides exactly the digit-level errors this project looks for.
- **Context 271.** The longest record is 269 characters; with its two newlines the model always sees a whole record at once.
- **The teacher phrases, Python counts.** Small LLMs misstate arithmetic. Every percentage is computed before the LLM sees the record, and every number it writes is checked against the record.
- **Rules first, the model second.** The model notices arithmetic errors on a given record but cannot rank them across records; a rule can. Rules take the arithmetic, the model ranks what rules cannot express.
- **Placeholders for numbers.** A small model writes plausible numbers rather than copying them; placing a placeholder is a task it can learn, and code guarantees the value.
- **A language model is not a rule engine.** Arithmetic relations such as Medicare ≤ total are cheap to check with a rule. The model's value is the errors no one wrote a rule for. The detection table shows which error types it catches and which still need a rule.

## Limitations

Two states, one year, and aggregated data. A planted error is a stand-in for a real one. A record from a hospital type the model has never seen can look surprising without being wrong, so the flagger ranks records for review; it does not decide.

## Credits

- Model, bigram baseline and training loop adapted from Kacper Raczy's course notebooks (`baseline.ipynb`, `transformer.ipynb`), which follow Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) lectures. Changes are marked in `src/model.py`.
- Teacher model: Qwen2.5-1.5B-Instruct (Qwen team, Alibaba), via Hugging Face `transformers`. Teacher–student training follows Hinton, Vinyals and Dean, *Distilling the Knowledge in a Neural Network*, 2015; the LLM-reference approach was Kacper Raczy's recommendation for this dataset.
- [Transformer Explainer](https://poloclub.github.io/transformer-explainer/) (Georgia Tech Polo Club), shared in the course, for the attention walkthrough.
- Wen et al., *Semantically Conditioned LSTM-based Natural Language Generation for Spoken Dialogue Systems*, EMNLP 2015 (delexicalisation).
- Nalisnick et al., *Do Deep Generative Models Know What They Don't Know?*, ICLR 2019; Ren et al., *Likelihood Ratios for Out-of-Distribution Detection*, NeurIPS 2019.
- Vaswani et al., *Attention Is All You Need*, 2017.
- Evaluation metrics as taught in the course (Lecture 6–7); BLEU and BERTScore through Hugging Face [`evaluate`](https://github.com/huggingface/evaluate).
- Data: Centers for Medicare & Medicaid Services.

AI assistance was used for code review and drafting; the design decisions, experiments and every reported number are from my own runs.

Cynthia Mutua · [cynthiamutua.com](https://cynthiamutua.com)
