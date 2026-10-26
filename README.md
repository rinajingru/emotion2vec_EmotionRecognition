# Task 1 - Speech Emotion Recognition

This project predicts the emotion expressed in one speech recording with the
official `emotion2vec+ seed` checkpoint.  It implements a complete working system as required by the midterm rubric: an audio file is passed through a
frozen emotion representation model and a lightweight classification head, and
the program returns the top emotion plus the complete score distribution.

## Quick start

Use Python 3.8 or newer.  Install the inference stack once:

```powershell
# 进入包含 ser_infer.py 和 requirements.txt 的目录后执行：
python -m pip install -r requirements.txt
```

Run the model on the bundled example audio that ships with the downloaded
checkpoint:

```powershell
python ser_infer.py --self-test --hub ms `
  --save-json outputs/demo_prediction.json
```

For your own recording:

```powershell
python ser_infer.py "C:\path\to\speech.wav" --hub ms `
  --save-json outputs/my_prediction.json
```

The first run downloads the checkpoint automatically.  The examples use
ModelScope (`ms`), which is convenient for mainland China; pass `--hub hf` for
Hugging Face instead.  Emotion2vec expects 16 kHz mono input. FunASR handles decoding and resampling; the script reports the expected sample rate (16000 Hz) in its JSON output. The default checkpoint predicts
nine categories: angry, disgusted, fearful, happy, neutral, other, sad,
surprised, and unknown.

`--extract-embedding` additionally requests the utterance representation from FunASR and records its shape in the JSON output. The original representation-only checkpoint from the paper can also be loaded by passing --model iic/emotion2vec.

## What the system does

Speech emotion recognition (SER) maps a variable-length speech recording to an
emotion label.  The input contains acoustic cues such as pitch, energy,
speaking rate, spectral shape, and voice quality; the output is a categorical
emotion distribution and its most likely class.  SER is useful in dialogue
systems, call-centre quality analysis, accessible interfaces, and affective
computing, but emotion is subjective and highly dependent on speaker, language,
context, and recording conditions.

The implementation follows Ma et al. (Findings of ACL 2024).  emotion2vec is
pre-trained with online teacher-student distillation on 262 hours of unlabeled
emotion speech.  Both networks use a seven-layer 1-D convolutional feature
extractor followed by a Transformer.  The student masks consecutive feature
frames and receives a learnable utterance token; the teacher remains unmasked
and is updated with an exponential moving average.  The pre-training objective
combines local frame-level MSE and global utterance-level MSE:

```
L = L_frame + alpha * L_utterance
L_frame      = mean over masked frames i of ||Y_i^T - Y_i^S||^2
L_utterance  = || mean over frames i of Y_i^T  -  U^S ||^2
```

Here Y^T and Y^S denote the teacher's and student's frame-level features, and U^S denotes the student's utterance-level representation produced by its learnable utterance token.

At inference time the encoder is frozen and `emotion2vec+ seed` supplies the
small downstream emotion classifier.  This preserves the paper's central
design: learn a reusable emotion representation once, then train only a small
task head.  The official FunASR interface returns `labels` and `scores`; the
script normalizes both dict and list return formats and writes a stable JSON
schema.

## Verification result

`outputs/demo_prediction.json` is produced by the `--self-test` command.  It
contains the downloaded checkpoint, the example audio path, the top prediction,
and all nine scores.  In the verified run, the 4.104-second example was
classified as `angry` with confidence `0.9999988`.  This is an end-to-end smoke
test rather than a benchmark accuracy claim; the exact score distribution is
kept in the generated JSON artifact.

The paper reports the following reference results for a frozen emotion2vec
representation with a linear downstream head (weighted accuracy, WA):

| Dataset | WA (%) | UA (%) | WF1 (%) |
| --- | ---: | ---: | ---: |
| IEMOCAP (4 classes) | 71.79 | - | - |
| MELD | 51.88 | 28.03 | 48.70 |
| RAVDESS | 82.43 | 82.86 | 82.39 |
| SAVEE | 84.38 | 82.30 | 84.45 |

These are paper results, not a claim that one arbitrary recording reaches the
same accuracy.  For an honest course evaluation, report both the paper table
and a held-out test set measured with WA, unweighted accuracy (UA), and weighted
F1 (WF1).

## Challenges, complexity, and limitations

* **Subjectivity and imbalance:** labels can be ambiguous and rare emotions can
  be underrepresented.  Report UA/WF1 in addition to overall accuracy and use
  class-weighted training for a local fine-tuning experiment.
* **Speaker, language, and channel shift:** a model may learn recording or
  speaker cues.  Split by speaker/session, standardize to mono 16 kHz, and
  evaluate at least one cross-domain set.
* **Noise and silence:** reverberation, clipping, and long pauses distort
  prosody.  Add VAD, loudness normalization, and realistic noise augmentation.
* **Long recordings and deployment cost:** the Transformer attention cost grows
  as `O(L^2 d)` for `L` downsampled frames.  Chunk long files, pool chunk
  predictions, and consider quantization or distillation for edge devices.

The released checkpoint is a strong zero-shot starting point, but it still
needs a downstream head for each target label inventory.  The paper also leaves
speaker-information removal and calibrated open-set confidence as useful next
steps.  A practical extension is a shared multi-task head with speaker-adversarial training, confidence calibration, and an explicit unknown class.

## Sources

* Ma, Ziyang et al. (2024), “emotion2vec: Self-Supervised Pre-Training for
  Speech Emotion Representation,” *Findings of ACL 2024*, pp. 15747-15760:
  <https://aclanthology.org/2024.findings-acl.931/>
* Official code and FunASR usage: <https://github.com/ddlBoJack/emotion2vec>
* `emotion2vec_plus_seed` model card:
  <https://huggingface.co/emotion2vec/emotion2vec_plus_seed>
