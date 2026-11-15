# How the code works

The project runs as one flow, from the CMS file to a flagged record. Each section below covers one step: what it does, why I built it that way, and where it lives in the code. Every stage runs from one command in `gpt.py`, and the stage headings there (`# ---- stage 1: data`) match the steps here.

## Step 1. Data — `src/data.py`

I read the CMS file, keep Michigan and Ohio, turn each row into one sentence, and write `train.txt`, `val.txt` and `test.txt` with one record per line. A language model learns from text, so each row becomes text. The newline between records marks the end of a record: the model learns that a record ends there, and generation stops when it writes one.

- `to_sentence`: the template. Dollar amounts are rounded to whole dollars, because cents add characters without adding meaning.
- `split_hospitals`: the split is by hospital, not by row. The 20 test hospitals never appear in training, so the test score measures whether the model learned what a record looks like, not whether it memorized one hospital's lines.
- `CharTokenizer`: one token per character, 77 in all. Characters the training data never contained (one `j` in the test set) are dropped.
- `get_batch`: picks random windows; the target is the same window shifted one character to the right. That shift is the whole training signal: predict the next character.

## Step 1b. LLM teacher — `src/teacher.py`

An LLM (Qwen2.5-1.5B-Instruct, on the Colab GPU) writes one plain-English sentence per record, and my small GPT later learns to write these sentences itself. This follows Kacper's suggestion from 3 November to have an LLM make the reference outputs. It is distillation: a large model teaches a small one.

- `facts`: Python computes every number, and each fact is handed over as a full phrase ("Medicare paid $14,681"), so the LLM copies the number together with its meaning. `SYSTEM` adds one invented example of the style.
- `check`: two tests. `numbers_in` rejects any number not in the facts (run 1: 15% of sentences). `wrong_label` rejects a real number next to the wrong word: run 1's teacher wrote "received $15,466 from Medicare" when $15,466 was the total. An LLM writing about money makes the same quiet mistakes the data can, so the teacher is checked like the data.
- A rejected record gets one retry with sampling; wrong twice, it is left out. Rejection counts go to `teacher_report.json`.
- Output line format: `record || sentence`. The ` || ` separator tells the student where the record ends and the explanation begins.

## Step 2. Models — `src/model.py`

A bigram baseline and a GPT, both adapted from Kacper's course notebooks; my changes are marked `change:`. The bigram comes first because it sees only the previous character: if the GPT can't beat it, the GPT has learned nothing.

The GPT, from the inside out:
- `Head`: one attention head. Query asks "what am I looking for", key says "what do I contain", value is "what I pass on". The scores are divided by the square root of the head size so the softmax doesn't saturate. The `tril` mask stops a character from looking ahead.
- `MultiHeadAttention`: four heads in parallel; each can follow a different pattern (the DRG code, the city).
- `FeedForward`: after attention, each position processes what it gathered. The hidden layer is 4× wider, as in GPT-2.
- `Block`: attention, then feed-forward, each with a residual connection (`x + ...`) and LayerNorm before it. Residuals keep gradients flowing; LayerNorm keeps training stable.
- `GPT`: character embedding plus position embedding, four blocks, a final LayerNorm, and a linear layer back to 77 scores, one per character.
- Context 271 = the longest record (269) plus two newlines, so the model always sees one whole record.

## Step 3. Training — `src/train.py`

AdamW, a learning rate that warms up and then follows a cosine curve down, weight decay, gradient clipping, early stopping, and a checkpoint that always holds the best model so far. My first prototype overfit: validation loss was best at step 1,500 (0.658) and rose to 0.676 while training loss kept falling. Each piece here is there because of that.
- `lr_at`: small steps while the weights are random, large steps in the middle, small steps at the end.
- `estimate_loss`: averages 50 batches so one lucky batch can't fool the log.
- The `patience` logic: if validation hasn't improved for 4 evaluations, stop.
- `on_eval`: perplexity and ROUGE-L (or record validity) are reported at every evaluation, from inside the loop. Kacper asked for this on 10 Nov.
- `save`: the checkpoint stores the weights, the architecture, the vocabulary and the training settings, so `--eval` can rebuild the exact model.

## Step 3b. Fine-tuning the explainer — `src/data.py`, `src/model.py`

My first student, trained from scratch on random windows, wrote sentences in the teacher's style about the wrong hospital, and none used the record's numbers. Random windows cut examples in half, and the loss rewarded predicting the record as much as the explanation. So I followed Lecture 6–7: pre-train, then fine-tune.
- `init_from` (model.py): the student starts from the record GPT, which already knows what records look like. Weights are matched by character, so new characters and the longer context start fresh.
- `encode_examples` and `get_sft_batch` (data.py): every row is one whole example starting at position 0, and the targets for the record part are -100, which `cross_entropy` skips. The model is graded only on the explanation it writes.
- Validation loss is therefore explanation-only; it is not comparable with run 1's.

The second student picked up the record's style and even its hospital system (for Munson Grayling it wrote about Munson Healthcare Otsego Memorial in Gaylord, a sister hospital), but none of its numbers were the record's. At under a million parameters, the model fills numbers from memory instead of copying digits.

## Step 3c. Placeholders — `delex` and `relex` in `src/data.py`

Before training, `delex` swaps the record's own values in the teacher's sentence for placeholders: `{BILL}`, `{TOTAL}`, `{SHARE}`, `{PATIENTS}`, `{HOSPITAL}`, `{CITY}`, `{CONDITION}`. The student learns which placeholder goes where, which is a pattern, not a memory. After generation, `relex` puts the record's real values back. This is delexicalisation (Wen et al., 2015), the standard way data-to-text systems keep numbers faithful.

The model decides the sentence and code guarantees the numbers. If the model ever writes a malformed placeholder, it shows up in the output instead of turning into a wrong number. With placeholders, 93% of the student's sentences pass the teacher's checks and all of them name the right hospital or city, against 0% and 2% without.

Testing on Illinois showed two facts still coming from memory: the state, which version 3 copied as a fixed "MI", and the condition whenever the teacher had rephrased it. Version 4 makes both placeholders as well: `CONDITION_SPAN` finds the teacher's condition phrase even when it isn't an exact match, and the state is matched as a whole word. `sentence_facts_ok` in `metrics.py` checks both, and `fully_correct` only counts a sentence when every check passes.

## Step 4. Generation — `src/generate.py`

Generation loads a checkpoint and writes one character at a time until it writes a newline.

- `idx[:, -model.context_size:]`: only the last 271 characters fit in the window.
- `temperature`: below 1 is safer and more repetitive; above 1 is more inventive.
- `top_k`: only the k most likely characters can be chosen.
- `torch.multinomial`: samples instead of always taking the top choice, which tends to loop.

## Step 5. Evaluation — `src/metrics.py`

| Metric | What it answers |
|---|---|
| Perplexity | How surprised the model is by unseen hospitals' records. Equal to e to the power of the loss. Lower is better. |
| Token accuracy | How often its top guess for the next character is right. |
| ROUGE-L, BLEU | Word overlap with the reference. Kacper calls these sanity checks: they rise early, then plateau. |
| BERTScore | Meaning overlap, so it gives credit for paraphrases. |
| Record validity | My own check: do generated records follow the rules real ones do (the right description for the DRG code, Medicare ≤ total payment)? |
| Numbers faithful | My own check: does every number in the student's sentence come from the record it was given? |

`validity_real_test_records` runs the same rules on real records. It is the ceiling: if a rule fails on real data, the rule is wrong, not the model.

## Step 6. Flagging — `src/flag.py`

I take clean records from unseen hospitals, plant one error per record, and check whether the model finds the corrupted versions more surprising than the clean ones. The four errors: a DRG code with another code's description, the hospital moved to another city in its state, the total and Medicare payments swapped, and covered charges given an extra zero.

Scoring: `char_surprise` (metrics.py) gives the model's surprise at every character. `scores` (flag.py) turns that into three numbers per record:
- `raw`: mean surprise over the whole record.
- `cond`: peak surprise over any 8 characters after the hospital name. In run 1 the unseen name was always the peak, so planted errors tied with clean records at exactly 0.5.
- `ratio`: the same peak, after subtracting the bigram's surprise character by character. In run 1 a wrong but common city looked more normal than a true but rare one (AUC 0.226). The bigram knows only how common characters are; what is left is what the GPT knows about context. This is a likelihood ratio (Ren et al., 2019).

For ranking I use ROC AUC: pick one corrupted and one clean record at random; how often does the corrupted one score higher? 0.5 is a coin flip, 1.0 is perfect, below 0.5 is backwards. `auc_ci` resamples 200 times for a 95% interval, so a difference of 0.02 is not mistaken for a finding.

What the runs showed:
- Mismatched diagnoses: 0.68 in every run. This is the model's real skill.
- Wrong city: 0.21, reliably backwards. Surprise tracks familiarity: a common wrong city looks more normal than a rare true one.
- Swapped payments and an extra digit: chance when *ranking* across records, but the paired test (run 3a) shows the model *notices* them on the same record 90% and 99% of the time. The signal is there; differences between hospitals drown it. A rule catches them every time, so rules go first.
- Run 2 tried two corrections, scoring after the name (`cond`) and the ratio to the bigram (`ratio`). Both made diagnosis detection worse (0.68 → 0.52), because the next rare thing after the name is the unseen hospital's city. I kept them in the code and the results, and `raw` stays the score.
- `paired_raw` asks the cleaner question: on the same record, does the error raise surprise? Diagnosis 0.85, payments 0.90, extra digit 0.99, wrong city 0.16.

So the tool is a hybrid. `flag_file` applies rules first (`rule_hits`: Medicare more than the total, paid more than billed, a bill beyond the 99.5th percentile of charge-to-payment ratios), then ranks the rest by the model's surprise. A wrong city needs a lookup against CMS's hospital register, not a language model.
