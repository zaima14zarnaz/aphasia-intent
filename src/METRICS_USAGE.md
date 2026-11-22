# Validation response metrics

Put `metrics.py` beside `train.py` and replace the trainer with the updated
version. The LoRA-only `model/pC.py`, formatter, and consolidated `losses.py`
pipeline remain in use. All losses still live in `losses.py`.

Every validation epoch computes all seven requested metric types, displays them
in the terminal, and appends one CSV row to:

```text
/home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/runs/metrics.csv
```

Override with `--metrics-csv /another/path/metrics.csv` when needed. Parent
directories are created. Existing rows are preserved; `run_id` distinguishes
experiments, and incompatible existing CSV headers cause an error instead of a
corrupted append. Per-epoch JSON history/checkpoints also contain the new scores.

## Install and prepare once

In your training environment:

```bash
python -m pip install -r requirements-metrics.txt
python -m nltk.downloader wordnet omw-1.4
```

BERTScore defaults to `roberta-large`, downloaded/cached by Hugging Face when
initialized. It runs on CPU by default. `--metrics-device cuda:0` enables GPU
scoring if you choose; `--metrics-batch-size 8` controls neural scoring batches.
For a local BERTScore encoder, pass `--bertscore-model /path/to/model` and
`--bertscore-num-layers N` for the desired encoder layer.

BLEURT uses Google's **official TensorFlow package** and the **BLEURT-20**
checkpoint. It runs in a separate CPU process. You can install it in the same
environment, or point `--bleurt-python` at a separate Python environment that
contains its TensorFlow/BLEURT dependencies. The separate environment does not
need Qwen, PEFT, or PyTorch. Follow the official installation instructions for a
TensorFlow version compatible with that environment.

Example installation using a separate environment:

```bash
python -m venv /path/to/bleurt-env
/path/to/bleurt-env/bin/python -m pip install --upgrade pip
/path/to/bleurt-env/bin/python -m pip install 'git+https://github.com/google-research/bleurt.git'
```

Download and extract the official checkpoint from your project root:

```bash
mkdir -p checkpoints
wget -O checkpoints/BLEURT-20.zip https://storage.googleapis.com/bleurt-oss-21/BLEURT-20.zip
unzip checkpoints/BLEURT-20.zip -d checkpoints
```

Then train:

```bash
python train.py --epochs 3 \
  --bleurt-checkpoint /home/z/zaimazarnaz/research1/Aphasia/src/intent_generation/checkpoints/BLEURT-20
```

If BLEURT is installed in the training environment, omit `--bleurt-python`.
`--bleurt-timeout 1800` sets the worker timeout in seconds. All metrics are
required by this trainer: missing packages, missing WordNet/checkpoint files,
or model-load failures stop setup before the training run. BLEURT is probed
with one sentence pair to check actual checkpoint inference. Failures preserve
the worker traceback; they are not silently replaced with zero or missing scores.

## What is reported

The two fields are evaluated separately: **intent** and **reasoning**. Evidence
is ignored. Scores are calculated on freely generated responses, not teacher-
forced token predictions. Every annotation remains a training reference.

For validation generation, repeated annotation rows with the same source clip
and transcript reuse one generated response. Their references are collected
together. Different transcripts for a clip remain different inputs; the report
shows both unique-input and unique-clip counts. Unique inputs have equal weight
in generation metrics, independent of the number of annotation references.

| Metric | Aggregation and scale |
| --- | --- |
| BERTScore F1 | Raw contextual F1, no baseline rescaling or IDF; best and mean reference scores, averaged across inputs |
| ROUGE-1/2/L | F1 with stemming; best and mean reference scores, averaged across inputs |
| METEOR | NLTK English METEOR with Treebank tokenization and WordNet; best and mean reference scores |
| BLEU | SacreBLEU multi-reference corpus score, **0–100**, tokenizer 13a, exponential smoothing, effective order enabled for short responses |
| BLEURT | Official checkpoint scores; best and mean reference scores; no clipping/rescaling |
| Required-field completion rate | Fraction of inputs with both nonempty intent and reasoning strings |
| Missing-field rate | Missing required-field slots divided by twice the number of inputs |

Also reported: missing-intent rate, missing-reasoning rate, unique inputs/clips,
reference count, and generated JSON validity. Rate columns in CSV are fractions
between 0 and 1; terminal rates display percentages. BLEURT may produce values
outside [0,1]. BERTScore/ROUGE/METEOR are not percentages.

Missing, empty, non-string or duplicated required fields follow the formatter's
missing-field policy. Malformed JSON counts both fields as missing. Text metrics
evaluate missing fields as empty candidates; raw BERTScore is zero for empty
candidates, while BLEURT uses its actual score for an empty candidate. Failed
responses are therefore retained, and every field's coverage is reported.

Column examples:

```text
val_intent_bertscore_f1_best
val_intent_bertscore_f1_mean
val_reasoning_rougeL_f1_best
val_reasoning_meteor_mean
val_intent_bleu
val_reasoning_bleurt_best
val_required_field_completion_rate
val_missing_field_rate
val_intent_missing_field_rate
val_reasoning_missing_field_rate
```

The CSV includes existing train/train-evaluation/validation losses and model
metadata. The BERTScore settings/hash, BLEURT checkpoint, and reference policy
are also saved in each run's `metrics_config.json`. Metric computations are
outside autograd and do not change the training loss or checkpoint selection.
Best checkpoint selection continues to use validation response loss.

## Verification

Checked real ROUGE, WordNet-backed METEOR, multi-reference SacreBLEU and BERTScore
using a tiny locally initialized encoder, including missing fields, reference
aggregation and CSV append behavior. A two-epoch tiny-Qwen LoRA run checked the
validation generation cache, terminal output and one CSV row per epoch. BLEURT's
CPU-worker protocol was checked with a test double; the pretrained BLEURT-20
checkpoint and full Qwen-7B weights were not run here. Production scoring calls
the official BLEURT package and requires its real checkpoint.

## Sources

- [BERTScore implementation](https://github.com/Tiiiger/bert_score)
- [ROUGE implementation](https://github.com/google-research/google-research/tree/master/rouge)
- [NLTK METEOR](https://www.nltk.org/api/nltk.translate.meteor_score.html)
- [SacreBLEU](https://github.com/mjpost/sacrebleu)
- [Official BLEURT setup and checkpoints](https://github.com/google-research/bleurt)
