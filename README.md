# ReProF: Residual Prototype Fusion for Continual LLM Attribution

## Introduction

Continual LLM attribution identifies the source model of generated text as new LLMs emerge over time. It is formulated as a class-incremental learning problem: at each stage, the system receives text from newly introduced source models, must identify every source encountered so far, and cannot retain historical training samples. Existing training-free prototype methods use a frozen feature extractor and analytic classifiers, but generally rely only on its final-layer representation. This discards complementary stylistic cues from other layers, while directly averaging or concatenating multi-layer features can mix useful signals with noise.

To address this problem, we propose ReProF (**Re**sidual **Pro**totype **F**usion), a training-free framework that exploits the hierarchy of a frozen feature extractor. ReProF partitions hidden layers into contiguous blocks, selects informative deep blocks, and constructs analytic prototypes for each block. The prototypes are fitted from shallow to deep: each deeper block models the residual left by preceding blocks. Their cumulative predictions combine complementary stylistic cues while retaining a rehearsal-free continual-learning pipeline. We evaluate ReProF on [COT-Bench](https://huggingface.co/datasets/sleep24/COT-bench/tree/main), a public benchmark for continual LLM attribution covering human-written text and 19 LLMs across eight domains. ReProF achieves a new state of the art on this benchmark, consistently outperforming existing methods across the evaluated domains.

<p align="center">
  <img src="resources/reprof_overview.png" alt="Overview of ReProF" width="100%">
</p>


## Requirements

### 🗂️ Environment

- `torch==2.1.0`
- `torchvision==0.16.0`
- `transformers==4.47.1`
- Other dependencies are listed in [`requirements.txt`](requirements.txt).

### 🤗 Pre-trained feature extractors

Download the desired checkpoints from Hugging Face before running an experiment. The code supports:

- [Llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B)
- [Llama-3-8B-Instruct](https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct)
- [Mistral-7B-Instruct-v0.3](https://huggingface.co/mistralai/Mistral-7B-Instruct-v0.3)
- [Qwen2.5-0.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
- [RoBERTa-large](https://huggingface.co/FacebookAI/roberta-large)

Set each local checkpoint path in `MODEL_REGISTRY` in `utils/inc_net.py`. For tokenization, use the corresponding local paths in `utils/data_manager.py` as well.

## 🔎 Dataset

[COT-Bench](https://huggingface.co/datasets/sleep24/COT-bench/tree/main) is a benchmark for continual LLM attribution. It covers eight domains—Abstracts, Books, News, Poetry, Recipes, Reddit, Reviews, and Wikipedia—and provides human-written text together with outputs from 19 LLMs released by 12 vendors.

The source LLMs are:

- **OpenAI:** GPT-2, GPT-3, ChatGPT, GPT-4, GPT-4o
- **Cohere:** Cohere, Cohere-Chat
- **MPT:** MPT, MPT-Chat
- **LLaMA:** LLaMA-Chat
- **Mistral:** Mistral, Mistral-Chat
- **Gemini:** Gemini-Pro
- **GLM:** GLM-4-Plus
- **Spark:** Spark-Max
- **ERNIE:** ERNIE-4.0-Turbo
- **Doubao:** Doubao-Pro
- **Qwen:** Qwen-Plus
- **DeepSeek:** Deepseek-Chat

Download COT-Bench from [Hugging Face](https://huggingface.co/datasets/sleep24/COT-bench/tree/main) or [Google Drive](https://drive.google.com/drive/folders/195dNDYPBV_dTzelu_foxhDhbVPy6VH-U?usp=drive_link), then place the `COT-Bench` directory under `./data/`.

## 🔑 Running scripts

The main ReProF configuration is `exps/ReProf.json`. After setting the dataset, checkpoint, and device paths, run:

```bash
python main.py --config exps/ReProf.json
```

For example, to run ReProF on the Abstracts domain, set `dataset` in `exps/ReProf.json` to your local `data/COT-Bench/abstracts` directory, then use the command above.

## 👨‍🏫 Acknowledgment

This codebase is built upon [LAMDA-PILOT](https://github.com/LAMDA-CL/LAMDA-PILOT) and the continual LLM attribution codebase from [GLPD](https://github.com/lhr01/continual-origin-tracing). We thank their authors for releasing these resources.
