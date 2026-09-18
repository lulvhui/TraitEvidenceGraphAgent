# TraitEvidenceGraphAgent

This repository contains the inference code for multimodal personality reasoning with **Qwen3-Omni-30B-A3B-Instruct + TraitEvidenceGraph**.

The current release supports:

- **IELTS**: dev and test splits
- **RecruitView**: dev and test splits
- **Multimodal input**:
  - turn-level transcript
  - video frames
  - synchronized audio embedded in the video
- **Output**:
  - O/C/E/A/N personality reasoning for each subject
  - corresponding debug evidence graphs

The source code is hosted on GitHub, while the media data are distributed separately through Hugging Face.

---

## 1. Clone the Repository

```bash
git clone https://github.com/lulvhui/TraitEvidenceGraphAgent.git
cd TraitEvidenceGraphAgent
```

The following instructions assume that the current working directory is the repository root:

```text
TraitEvidenceGraphAgent/
```

---

## 2. Repository Structure

The expected directory structure is:

```text
TraitEvidenceGraphAgent/
├── bridges/
├── configs/
│   └── qwen3_omni_local.yaml
├── inputs/
│   ├── IELTS/
│   │   ├── dev.input.jsonl
│   │   └── test.input.jsonl
│   └── RecruitView/
│       ├── dev.input.jsonl
│       └── test.input.jsonl
├── manifests/
├── models/
│   ├── Qwen3-Omni-30B-A3B-Instruct/
│   └── bge-m3/
├── data/
│   ├── IELTS/
│   └── RecruitView/
├── outputs/
├── scripts/
├── src/
├── requirements.txt
└── requirements-optional-nf4.txt
```

The following directories are not included directly in the GitHub repository and need to be prepared on the target server:

```text
models/
data/
outputs/
```

---

## 3. Model Setup

### 3.1 Qwen3-Omni-30B-A3B-Instruct

The main backbone used in this project is:

```text
Qwen/Qwen3-Omni-30B-A3B-Instruct
```

The model is assumed to be already available on the target server.

By default, the configuration expects the model at:

```text
models/Qwen3-Omni-30B-A3B-Instruct
```

The corresponding configuration is:

```yaml
backbone:
  model_path: models/Qwen3-Omni-30B-A3B-Instruct
```

If the model is stored elsewhere on the server, either modify `model_path` in:

```text
configs/qwen3_omni_local.yaml
```

or create a symbolic link:

```bash
mkdir -p models

ln -s /PATH/TO/Qwen3-Omni-30B-A3B-Instruct \
  models/Qwen3-Omni-30B-A3B-Instruct
```

Official resources:

- Model card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Official repository: https://github.com/QwenLM/Qwen3-Omni

### 3.2 BGE-M3

TraitEvidenceGraph uses **BGE-M3** to compute cross-context semantic similarity.

The default path is:

```text
models/bge-m3
```

If BGE-M3 is not already available on the server, install the Hugging Face CLI and download it:

```bash
pip install -U huggingface_hub

mkdir -p models

hf download \
  BAAI/bge-m3 \
  --local-dir models/bge-m3
```

If BGE-M3 already exists elsewhere on the server, either update:

```yaml
context_similarity:
  model_path: models/bge-m3
```

or create a symbolic link.

---

## 4. Dataset Setup

The multimodal data are hosted on Hugging Face:

```text
lulvhui/TraitEvidenceGraphAgent
```

The four input JSONL files required for inference are already included in the GitHub repository:

```text
inputs/IELTS/dev.input.jsonl
inputs/IELTS/test.input.jsonl
inputs/RecruitView/dev.input.jsonl
inputs/RecruitView/test.input.jsonl
```

The media files need to be downloaded separately.

### 4.1 Download the Media Data

Install the Hugging Face CLI if necessary:

```bash
pip install -U huggingface_hub
```

If the dataset repository requires authentication:

```bash
hf auth login
```

Download the dataset into the repository's `data/` directory:

```bash
hf download \
  lulvhui/TraitEvidenceGraphAgent \
  --repo-type dataset \
  --local-dir data
```

After downloading, the required media paths should follow the structure:

```text
data/
├── IELTS/
│   └── processed/
│       └── video_turn_clips/
│           └── ...
└── RecruitView/
    └── raw/
        └── videos/
            └── ...
```

The paths must match the `video_path` fields in the corresponding input JSONL files.

For example:

```text
data/IELTS/processed/video_turn_clips/04/turn_001.mp4
```

### 4.2 Dataset Statistics

| Dataset | Split | People | Turns / MP4 | Media Size |
|---|---:|---:|---:|---:|
| IELTS | dev | 33 | 357 | 6.39 GB |
| IELTS | test | 33 | 346 | 7.07 GB |
| RecruitView | dev | 47 | 289 | 2.31 GB |
| RecruitView | test | 47 | 310 | 2.51 GB |
| Total |  | 160 split-person entries | 1302 unique MP4 | 18.28 GB |

The exact media file lists are provided in:

```text
manifests/IELTS_dev_media.txt
manifests/IELTS_test_media.txt
manifests/RecruitView_dev_media.txt
manifests/RecruitView_test_media.txt
manifests/all_dev_test_media.txt
```

Audio is already embedded in the MP4 audio tracks. No separate WAV files are required.

---

## 5. Environment Setup

Create a dedicated Conda environment:

```bash
conda create -n qwen3omni_traitgraph python=3.11 -y
conda activate qwen3omni_traitgraph

cd TraitEvidenceGraphAgent
```

Install a CUDA-compatible PyTorch build according to the CUDA driver available on the target server.

For example:

```bash
pip install torch torchvision torchaudio
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Install FlashAttention 2:

```bash
pip install flash-attn --no-build-isolation
```

FFmpeg is also required:

```bash
conda install -c conda-forge ffmpeg -y
```

The project requires:

```text
transformers >= 5.2.0, < 6
```

The required Qwen3-Omni classes include:

```text
Qwen3OmniMoeForConditionalGeneration
Qwen3OmniMoeProcessor
```

---

## 6. GPU and Precision Configuration

The default configuration uses full BF16 weights:

```yaml
load_in_4bit: false
dtype: bfloat16
device_map: auto
attn_implementation: flash_attention_2
```

Full-BF16 inference requires a high-memory multi-GPU environment.

The model uses:

```yaml
device_map: auto
```

so model components are automatically distributed across the GPUs visible to the current process.

The Talker component is disabled after loading because the current task only requires text generation.

### Optional NF4 Mode

If the available GPU memory is insufficient for full BF16 inference, install the optional quantization dependencies:

```bash
pip install -r requirements-optional-nf4.txt
```

Then modify:

```text
configs/qwen3_omni_local.yaml
```

from:

```yaml
load_in_4bit: false
```

to:

```yaml
load_in_4bit: true
```

NF4 inference should be treated as a separate quantized configuration rather than the default BF16 experiment.

### FlashAttention Fallback

The default configuration uses:

```yaml
attn_implementation: flash_attention_2
```

If FlashAttention 2 is not supported on the target server, change it to:

```yaml
attn_implementation: sdpa
```

`sdpa` is generally slower and may require more GPU memory.

---

## 7. Preflight Check

Before starting inference, validate the environment, models, input files, and media paths.

Run:

```bash
python scripts/preflight.py --dataset IELTS --split dev
python scripts/preflight.py --dataset IELTS --split test

python scripts/preflight.py --dataset RecruitView --split dev
python scripts/preflight.py --dataset RecruitView --split test
```

Each command should return:

```json
{
  "valid": true
}
```

The preflight script checks:

- Qwen3-Omni model directory
- model type
- BGE-M3 directory
- Transformers version
- CUDA availability
- available GPU memory
- FlashAttention installation
- FFmpeg
- input JSONL schema
- dataset identity
- existence of every referenced MP4 file

Do not start the full inference run until the required preflight checks pass.

---

## 8. Full Evaluation Workflow

For reproducing the complete evaluation, the recommended execution order is:

1. IELTS dev
2. RecruitView dev
3. verify the dev outputs
4. IELTS test
5. RecruitView test

This avoids changing the configuration after inspecting test results.

### 8.1 Run All Four Splits

For example, when using physical GPUs `0,1,2,3`:

```bash
export PYTHON_BIN=$(which python)

bash scripts/run_both_datasets.sh 0,1,2,3
```

The script runs:

```text
IELTS dev
    ↓
RecruitView dev
    ↓
IELTS test
    ↓
RecruitView test
```

Each split supports automatic resume after interruption.

### 8.2 Run Individual Splits

IELTS dev:

```bash
bash scripts/run_split.sh IELTS dev 0,1,2,3
```

RecruitView dev:

```bash
bash scripts/run_split.sh RecruitView dev 0,1,2,3
```

IELTS test:

```bash
bash scripts/run_split.sh IELTS test 0,1,2,3
```

RecruitView test:

```bash
bash scripts/run_split.sh RecruitView test 0,1,2,3
```

If different GPU IDs should be used, modify the final argument.

For example:

```bash
bash scripts/run_split.sh RecruitView dev 2,3,4,5
```

The script internally sets:

```bash
CUDA_VISIBLE_DEVICES=<GPU_IDS>
```

---

## 9. Output Files

The inference outputs are written to:

```text
outputs/
├── IELTS/
│   └── qwen3_omni_30b_a3b/
│       └── trait_evidence_graph/
│           ├── dev/
│           │   ├── result.json
│           │   └── debug.json
│           ├── test/
│           │   ├── result.json
│           │   └── debug.json
│           └── logs/
│               ├── dev.log
│               └── test.log
│
└── RecruitView/
    └── qwen3_omni_30b_a3b/
        └── trait_evidence_graph/
            ├── dev/
            │   ├── result.json
            │   └── debug.json
            ├── test/
            │   ├── result.json
            │   └── debug.json
            └── logs/
                ├── dev.log
                └── test.log
```

Specifically:

```text
outputs/IELTS/qwen3_omni_30b_a3b/trait_evidence_graph/dev/result.json
outputs/IELTS/qwen3_omni_30b_a3b/trait_evidence_graph/dev/debug.json

outputs/IELTS/qwen3_omni_30b_a3b/trait_evidence_graph/test/result.json
outputs/IELTS/qwen3_omni_30b_a3b/trait_evidence_graph/test/debug.json

outputs/RecruitView/qwen3_omni_30b_a3b/trait_evidence_graph/dev/result.json
outputs/RecruitView/qwen3_omni_30b_a3b/trait_evidence_graph/dev/debug.json

outputs/RecruitView/qwen3_omni_30b_a3b/trait_evidence_graph/test/result.json
outputs/RecruitView/qwen3_omni_30b_a3b/trait_evidence_graph/test/debug.json
```

Each completed split is automatically validated.

The validation checks that:

- input and output subject identities match
- every subject contains O/C/E/A/N outputs
- every reasoning field is non-empty
- both `result.json` and `debug.json` cover the complete split

---

## 10. Default Multimodal Inference Configuration

The released configuration uses the following fixed multimodal settings:

```text
video fps:                 0.5
per-frame pixels:          50176
per-video total pixels:    524288
audio sample rate:         16000
audio max duration/turn:   16 seconds
generation max tokens:     384
sampling:                  disabled
```

The corresponding settings are defined in:

```text
configs/qwen3_omni_local.yaml
```

Important configuration values include:

```yaml
audio:
  sample_rate: 16000
  max_seconds_per_input: 16.0

video:
  fps: 0.5
  min_pixels: 50176
  max_pixels: 50176
  total_pixels: 524288

generation:
  max_new_tokens: 384
  do_sample: false
```

For reproducible evaluation, keep these settings unchanged unless intentionally testing a different configuration.

---

## 11. Resume Interrupted Runs

`run_split.sh` always enables:

```text
--resume
```

If inference is interrupted, run the same command again.

For example:

```bash
bash scripts/run_split.sh IELTS test 0,1,2,3
```

Already completed subjects whose results have been successfully written to both the result and debug outputs will be preserved.

Inference will continue from the remaining subjects.

Do not manually concatenate partial JSON files.

---

## 12. Common Issues

### Model directory not found

If preflight reports:

```text
Missing Qwen3-Omni model directory
```

check:

```yaml
backbone:
  model_path: ...
```

in:

```text
configs/qwen3_omni_local.yaml
```

and make sure it points to the existing Qwen3-Omni model directory.

---

### BGE-M3 directory not found

Check:

```yaml
context_similarity:
  model_path: models/bge-m3
```

and verify that `config.json` exists inside the BGE-M3 directory.

---

### Missing media files

If preflight reports:

```text
Missing media files
```

verify that the Hugging Face dataset has been downloaded under:

```text
data/
```

and that paths such as:

```text
data/IELTS/processed/video_turn_clips/...
data/RecruitView/raw/videos/...
```

exist exactly as referenced by the input JSONL files.

---

### FlashAttention is unavailable

Either install:

```bash
pip install flash-attn --no-build-isolation
```

or change:

```yaml
attn_implementation: flash_attention_2
```

to:

```yaml
attn_implementation: sdpa
```

---

### Out of GPU memory

Possible options include:

1. use GPUs with more available memory
2. expose additional GPUs to the process
3. ensure no unrelated processes are occupying GPU memory
4. use the optional NF4 configuration

Check current GPU usage with:

```bash
nvidia-smi
```

---

## 13. Quick Start

After the repository, models, dataset, and environment have been prepared:

```bash
conda activate qwen3omni_traitgraph

cd TraitEvidenceGraphAgent
```

Check one split first:

```bash
python scripts/preflight.py \
  --dataset IELTS \
  --split dev
```

If the output contains:

```json
{
  "valid": true
}
```

run a single dev split:

```bash
bash scripts/run_split.sh IELTS dev 0,1,2,3
```

After confirming that the environment and outputs are correct, run the complete evaluation:

```bash
bash scripts/run_both_datasets.sh 0,1,2,3
```
