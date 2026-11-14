# Walkthrough: how the code flows

Read the files in this order. Each section says what the step does, why it is built that way, and which lines to look at. Every stage runs from one command in `gpt.py`; the stage headings in `gpt.py` (`# ---- stage 1: data`) match the steps below.

## Step 1. Data — `src/data.py`

**What.** Reads the CMS file, keeps Michigan and Ohio, turns each row into one sentence, and writes `train.txt`, `val.txt` and `test.txt` with one record per line.

**Why.** A language model learns from text, so each row becomes text. The newline between records is the end-of-record marker: the model learns that a record ends there, and generation stops when it writes one.

**Look at.**
- `to_sentence`: the template. Dollar amounts are rounded to whole dollars, because cents add characters without adding meaning.
- `split_hospitals`: the split is by hospital, not by row. The 20 test hospitals never appear in training, so the test score measures whether the model learned what a record looks like, not whether it memorized one hospital's lines.
- `CharTokenizer`: one token per character, 77 in all. Characters the training data never contained (one `j` in the test set) are dropped.
- `get_batch`: picks random windows; the target is the same window shifted one character to the right. That shift is the whole training signal: predict the next character.

## Step 1b. LLM teacher — `src/teacher.py`

**What.** An LLM (Qwen2.5-1.5B-Instruct, on the Colab GPU) writes one plain-English sentence per record. The small GPT later learns to write these sentences itself.

**Why.** Kacper's suggestion (3 Nov): use an LLM to make reference outputs, then train a small model to reproduce them. This is distillation: a large model teaches a small one.

**Look at.**
- `facts`: Python computes every number (Medicare's share of the payment, the share of the bill that was paid). The LLM only words the sentence.
- `check`: rejects any sentence that contains a number not in the facts. An LLM writing about money makes the same quiet mistakes the data can, so the teacher is checked too. The rejection count is reported.
- Output line format: `record || sentence`. The ` || ` separator tells the student where the record ends and the explanation begins.

## Step 2. Models — `src/model.py`

**What.** A bigram baseline and a GPT. Both are adapted from Kacper's notebooks; my changes are marked `change:`.

**Why a bigram first.** It sees only the previous character. If the GPT can't beat it, the GPT has learned nothing.

**The GPT, from the inside out.**
- `Head`: one attention head. Query asks "what am I looking for", key says "what do I contain", value is "what I pass on". The scores are divided by the square root of the head size so the softmax doesn't saturate. The `tril` mask stops a character from looking ahead.
- `MultiHeadAttention`: four heads in parallel; each can follow a different pattern (the DRG code, the city).
- `FeedForward`: after attention, each position processes what it gathered. The hidden layer is 4× wider, as in GPT-2.
- `Block`: attention, then feed-forward, each with a residual connection (`x + ...`) and LayerNorm before it. Residuals keep gradients flowing; LayerNorm keeps training stable.
- `GPT`: character embedding plus position embedding, four blocks, a final LayerNorm, and a linear layer back to 77 scores, one per character.
- Context 271 = the longest record (269) plus two newlines, so the model always sees one whole record.

## Step 3. Training — `src/train.py`

**What.** AdamW, a learning rate that warms up and then follows a cosine curve down, weight decay, gradient clipping, early stopping, and a checkpoint that always holds the best model so far.

**Why.** The first run (14 Nov) overfit: validation loss was best at step 1,500 (0.658) and rose to 0.676 while training loss kept falling. Each piece here answers that.
- `lr_at`: small steps while the weights are random, large steps in the middle, small steps at the end.
- `estimate_loss`: averages 50 batches so one lucky batch can't fool the log.
- The `patience` logic: if validation hasn't improved for 4 evaluations, stop.
- `on_eval`: perplexity and ROUGE-L (or record validity) are reported at every evaluation, from inside the loop. Kacper asked for this on 10 Nov.
- `save`: the checkpoint stores the weights, the architecture, the vocabulary and the training settings, so `--eval` can rebuild the exact model.

## Step 4. Generation — `src/generate.py`

**What.** Loads a checkpoint and writes one character at a time until it writes a newline.

**Look at.**
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
| Record validity | Mine: do generated records follow the rules real ones do (the right description for the DRG code, Medicare ≤ total payment)? |
| Numbers faithful | Mine: does every number in the student's sentence come from the record it was given? |

`validity_real_test_records` runs the same rules on real records. It is the ceiling: if a rule fails on real data, the rule is wrong, not the model.

## Step 6. Flagging — `src/flag.py`

**What.** Takes clean records from unseen hospitals, plants one error per record, and asks whether the model finds the corrupted versions more surprising than the clean ones.

**The four errors.** A DRG code with another code's description; the hospital moved to another city in its state; total and Medicare payments swapped; covered charges given an extra zero.

**The score.** `record_scores` in `metrics.py` returns two numbers per record: the mean surprise over the whole line, and the peak surprise over any 8-character stretch. One wrong field shows up in the peak.

**The result.** ROC AUC: pick one corrupted and one clean record at random; how often does the corrupted one score higher? 0.5 is a coin flip; 1.0 is perfect.

**What to expect, honestly.** The model should catch mismatches between words (code and description) better than arithmetic (the swapped payments), which a one-line rule catches anyway. Whichever way it comes out is the finding.

## Questions to expect, and the answers

- **Why characters, not BPE?** Records are short, and the errors live in single digits. A subword tokenizer can merge `$52` into one token and `24` into another, which hides a one-digit change.
- **Why split by hospital?** A row split puts the same hospital's name and address in both train and test, so the test score would measure memory.
- **Why did the first run overfit?** 0.85M parameters on 1.6M characters, a constant learning rate, and no stopping rule. Early stopping, weight decay and the cosine schedule answer it; the hyperparameter table shows the effect of model size.
- **Can the teacher's mistakes leak into the student?** They can, which is why every teacher number is computed in Python and checked before training, and the student is checked the same way.
- **Is the flagger a replacement for rules?** No. Rules catch what someone thought to write down. The model ranks what no one wrote a rule for, for a person to review.
- **What would you do next?** More states and years, a KV cache for faster generation, and testing the flagger on real errors, not planted ones.
