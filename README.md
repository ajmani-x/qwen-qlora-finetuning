# QLoRA Fine-Tuning of Qwen2.5-1.5B-Instruct

A parameter-efficient fine-tuning experiment using **QLoRA**, **Hugging Face Transformers**, and **PEFT** to adapt Qwen2.5-1.5B-Instruct for concise, beginner-friendly explanations of AI and machine learning concepts.

The project covers the complete workflow, from loading a 4-bit quantized language model and preparing instruction data to training LoRA adapters, evaluating loss, and saving the resulting adapters.

## Overview

Full fine-tuning of large language models can require substantial GPU memory. QLoRA reduces memory requirements by keeping the pretrained model in 4-bit precision while training a relatively small set of LoRA adapter parameters.

In this experiment, I fine-tuned **Qwen/Qwen2.5-1.5B-Instruct** on a small instruction-following dataset containing introductory AI and machine learning questions. The intended response style was concise, accessible, and supported by practical examples where appropriate.

This is a proof of concept focused on understanding and implementing the fine-tuning pipeline, rather than a production-ready model.

## Tech Stack

- **Base model:** Qwen2.5-1.5B-Instruct
- **Fine-tuning:** QLoRA, PEFT
- **Libraries:** PyTorch, Transformers, Datasets, BitsAndBytes, Accelerate
- **Training:** Hugging Face `Trainer`
- **Environment:** Google Colab with NVIDIA T4 GPU

## Implementation

### 1. Quantized model loading

Loaded the pretrained model with **4-bit NF4 quantization**, double quantization, and BitsAndBytes to reduce GPU memory usage.

### 2. LoRA configuration

Attached LoRA adapters to the attention and feed-forward projection layers using the following configuration:

| Parameter | Value |
|---|---|
| Rank (`r`) | 16 |
| LoRA alpha | 32 |
| LoRA dropout | 0.05 |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable parameters | 18,464,768 |
| Trainable share | 1.1820% |

The pretrained model parameters remained frozen while the LoRA adapter parameters were updated.

### 3. Dataset preparation

Created a small conversational dataset of AI and machine learning questions paired with short, beginner-friendly answers. The dataset was split into **16 training examples** and **4 validation examples**.

Training labels were prepared to calculate loss on assistant completions rather than the instruction prompt.

### 4. Training

Fine-tuned the adapters for **3 epochs** using the standard Hugging Face `Trainer`. The successful training run used `fp16=False`, `bf16=False`, and `gradient_checkpointing=False` after encountering compatibility issues with an optimized TRL loss path and mixed-precision gradient scaling.

## Training Results

| Metric | Result |
|---|---:|
| Training examples | 16 |
| Validation examples | 4 |
| Epochs | 3 |
| Optimization steps | 12 |
| Training runtime | 50.75 seconds |
| Total parameters reported | 1,562,179,072 |
| Trainable parameters | 18,464,768 (1.1820%) |

| Epoch | Training Loss | Validation Loss |
|---|---:|---:|
| 1 | 1.673355 | 1.435578 |
| 2 | 1.155806 | 1.333263 |
| 3 | 1.128460 | 1.332811 |

Both training and validation loss decreased during the run. However, the validation set contained only four examples, so these results should be interpreted as an initial pipeline check rather than evidence of broad generalization.

## Evaluation and Observations

I compared model responses before and after fine-tuning on several introductory questions. Some responses became shorter, but others remained verbose, indicating that the small dataset did not consistently establish the intended response style.

The preliminary comparison also used different prompting conditions from the training setup. A more reliable assessment would compare the base and fine-tuned models using identical system prompts, generation settings, and a separate held-out test set.

## Repository Structure

```text
qwen-qlora-finetuning/
├── QLoRA_FineTuning.ipynb
├── README.md
├── requirements.txt
└── .gitignore
```

## Running the Notebook

1. Open `QLoRA_FineTuning.ipynb` in [Google Colab](https://colab.research.google.com/).
2. Enable a CUDA GPU runtime (the experiment was conducted on an NVIDIA T4).
3. Run the notebook cells sequentially to install dependencies, load the model, prepare the dataset, and train the adapters.
4. Save the trained LoRA adapters locally or copy them to Google Drive for persistence.

**Reproducibility note:** The original environment's exact package versions were not recorded, and the cleaned notebook has not been independently rerun in a fresh runtime. Changes in Transformers, PEFT, or related libraries may require minor compatibility adjustments.

## Limitations and Future Improvements

- Expand the dataset with more diverse, carefully reviewed examples.
- Use separate training, validation, and test sets.
- Evaluate answer length, clarity, factual accuracy, and instruction-following under matched prompts.
- Record and pin dependency versions for reproducible training.
- Compare different LoRA ranks, learning rates, and training durations.

## Model and Weights

The base model is available on [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-1.5B-Instruct). The fine-tuned LoRA adapters were saved separately to Google Drive and are not included in this repository.
