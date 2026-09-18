# Qwen3-Omni-30B-A3B TraitEvidenceGraph 便携推理包

该目录是在其他服务器上运行 **Qwen3-Omni-30B-A3B-Instruct +
TraitEvidenceGraph** 所需的最小推理快照，覆盖：

- IELTS：dev 和 test；
- RecruitView：dev 和 test；
- 输入模态：Turn transcript + 原始视频帧 + 视频内同步音频；
- 输出：每位受试者的 O/C/E/A/N 五项 reasoning，以及单独的 debug graph。


## 1. 型号名称

本实验使用的官方开放权重名称是：

```text
Qwen/Qwen3-Omni-30B-A3B-Instruct
```


官方资源：

- Model card: https://huggingface.co/Qwen/Qwen3-Omni-30B-A3B-Instruct
- Official repository: https://github.com/QwenLM/Qwen3-Omni

## 2. 目录要求

最终目录应当是：

```text
qwen3_omni_30b_a3b_traitgraph/
├── bridges/
├── configs/
├── inputs/
│   ├── IELTS/{dev,test}.input.jsonl
│   └── RecruitView/{dev,test}.input.jsonl
├── manifests/
├── models/
│   ├── Qwen3-Omni-30B-A3B-Instruct/
│   └── bge-m3/
├── data/
│   ├── IELTS/processed/video_turn_clips/...
│   └── RecruitView/raw/videos/...
├── outputs/
├── scripts/
└── src/
```

`models/`、`data/` 和 `outputs/` 初始不存在，需要下载/迁移数据后创建。

## 3. 需要下载的模型

### 3.1 Qwen3-Omni-30B-A3B-Instruct


### 3.2 BGE-M3

Evidence graph 使用 BGE-M3 计算跨上下文相似度：

```bash
huggingface-cli download BAAI/bge-m3 --local-dir models/bge-m3
```


## 4. 需要迁移的数据

四个输入 JSONL 已包含在本目录中，但不包含视频。需要准备以下媒体：

| Dataset | Split | People | Turns / MP4 | 源服务器大小 |
|---|---:|---:|---:|---:|
| IELTS | dev | 33 | 357 | 6.39 GB |
| IELTS | test | 33 | 346 | 7.07 GB |
| RecruitView | dev | 47 | 289 | 2.31 GB |
| RecruitView | test | 47 | 310 | 2.51 GB |
| 合计 | | 160 split-person entries | 1302 unique MP4 | 18.28 GB |

准确文件列表：

```text
manifests/IELTS_dev_media.txt
manifests/IELTS_test_media.txt
manifests/RecruitView_dev_media.txt
manifests/RecruitView_test_media.txt
manifests/all_dev_test_media.txt
```

音频已经包含在 MP4 音轨中，不需要单独下载 WAV。仅下载原始公开数据但不按上述
相对路径放置是不够的；推理输入会严格检查每个文件。


```bash
cd /mnt/sda/lhlu/TraitEvidenceGraphAgent

rsync -av --files-from=deploy/qwen3_omni_30b_a3b_traitgraph/manifests/all_dev_test_media.txt \
  ./ \
  USER@NEW_SERVER:/PATH/qwen3_omni_30b_a3b_traitgraph/
```

这条命令会保留 `data/IELTS/...` 和 `data/RecruitView/...` 的相对目录结构。


## 5. 环境安装

建议新建独立环境：

```bash
conda create -n qwen3omni_traitgraph python=3.11 -y
conda activate qwen3omni_traitgraph
cd /PATH/qwen3_omni_30b_a3b_traitgraph
```

先根据服务器 CUDA/driver 安装匹配的 PyTorch，再安装其余依赖：

```bash
# 请从 https://pytorch.org/get-started/locally/ 选择与服务器匹配的命令
pip install torch torchvision torchaudio

pip install -r requirements.txt
pip install -U flash-attn --no-build-isolation
```

系统还必须安装 FFmpeg，例如：

```bash
conda install -c conda-forge ffmpeg -y
```

官方推荐 Transformers 5.2.0+；该版本包含
`Qwen3OmniMoeForConditionalGeneration` 和 `Qwen3OmniMoeProcessor`。

## 6. GPU 与精度

主配置使用完整 BF16 权重：

```yaml
load_in_4bit: false
dtype: bfloat16
device_map: auto
attn_implementation: flash_attention_2
```

建议至少使用 2×80 GB，或具有约 96 GB 以上总显存且单卡分配合理的多 GPU
服务器。Talker 在加载后关闭，仅生成文本；官方说明关闭 Talker 可节约约 10 GB
显存。Transformers 上的 MoE 推理仍可能很慢，这是正常现象。

如果机器无法容纳 BF16，可以安装：

```bash
pip install -r requirements-optional-nf4.txt
```

然后将 `configs/qwen3_omni_local.yaml` 中 `load_in_4bit` 改为 `true`。

如果服务器不支持 FlashAttention 2，可以明确把
`attn_implementation: flash_attention_2` 改为 `sdpa`；这通常更慢且占用更多显存。

## 7. 运行前检查

先检查四个 split：

```bash
python scripts/preflight.py --dataset IELTS --split dev
python scripts/preflight.py --dataset IELTS --split test
python scripts/preflight.py --dataset RecruitView --split dev
python scripts/preflight.py --dataset RecruitView --split test
```

四次都必须返回：

```json
{"valid": true}
```

检查内容包括模型类型、BGE-M3、Transformers 版本、CUDA、FlashAttention、
FFmpeg、JSONL schema 和每个 MP4 是否存在。

## 8. 正确实验顺序：dev 和 test 都必须推理

实验必须包括两个数据集的验证集和测试集。顺序固定为：

1. IELTS dev；
2. RecruitView dev；
3. 检查 dev 输出并冻结 config；
4. IELTS test；
5. RecruitView test。



### 8.1 一条命令依次运行全部四项

假设使用物理 GPU 0、1、2、3：

```bash
export PYTHON_BIN=$(which python)
bash scripts/run_both_datasets.sh 0,1,2,3
```

脚本会先完成两个 dev，验证输出完整后才开始两个 test。每个 split 都支持断点续跑。

### 8.2 分开运行

```bash
bash scripts/run_split.sh IELTS dev 0,1,2,3
bash scripts/run_split.sh RecruitView dev 0,1,2,3

# 确认 dev 后冻结配置，再执行 test：
bash scripts/run_split.sh IELTS test 0,1,2,3
bash scripts/run_split.sh RecruitView test 0,1,2,3
```

如果服务器 GPU 编号不同，修改最后一个参数即可，例如：

```bash
bash scripts/run_split.sh RecruitView dev 2,3,4,5
```

## 9. 输出位置

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

日志位于每个数据集的：

```text
outputs/<DATASET>/qwen3_omni_30b_a3b/trait_evidence_graph/logs/<SPLIT>.log
```

每个 split 结束后，脚本会确认：

- 输入与输出 person identity 完全一致；
- 每人都有 O/C/E/A/N；
- 所有 reasoning 非空；
- result 和 debug 都覆盖完整 split。

## 10. 固定的多模态预算

为便于与已有 Qwen2.5-Omni-7B 实验比较，本配置固定为：

```text
video fps:                 0.5
per-frame pixels:          50176
per-video total pixels:    524288
audio sample rate:         16000
audio max duration/turn:   16 seconds
generation max tokens:     384
sampling:                  disabled
```


## 11. 断点恢复

`run_split.sh` 内部始终传入 `--resume`。中断后执行同一命令即可，它会保留已经同时
写入 result/debug 的受试者，并从剩余受试者继续。不要手工拼接 JSON 文件。
