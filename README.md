# NoMIRACL: Knowing When You Don’t Know for Robust Multilingual Retrieval-Augmented Generation

[![GitHub stars](https://img.shields.io/github/stars/project-miracl/nomiracl.svg?style=flat&logo=github&colorB=blue&label=stars)](https://github.com/project-miracl/nomiracl)
[![GitHub license](https://img.shields.io/github/license/project-miracl/nomiracl.svg?style=flat&colorB=blue)](https://github.com/project-miracl/nomiracl/blob/main/LICENSE)

## Introduction

Retrieval Augmented Generation (RAG) is a powerful approach to incorporate external knowledge into large language models (LLMs) to enhance the accuracy and faithfulness of generated responses. However, evaluating LLM robustness in RAG across different language families has been a challenge, leading to gaps in understanding the model's performance against errors in external retrieved knowledge. To address this, we present NoMIRACL, a human-annotated dataset designed for evaluating LLM robustness in RAG across 18 typologically diverse languages.

NoMIRACL includes both a non-relevant and a relevant subset. The non-relevant subset contains queries with passages manually judged as non-relevant or noisy, while the relevant subset includes queries with at least one judged relevant passage. LLM robustness is measured using two key metrics: hallucination rate and error rate.

We provide a GPT-4 baseline trained on NoMIRACL, achieving a 33.2% hallucination rate on the non-relevant subset and a 14.9% error rate on the relevant subset on average. The evaluation highlights areas where GPT-4 tends to hallucinate frequently, emphasizing the need for future research to enhance LLM robustness.

## NoMIRACL Dataset and GPT-4 Baseline

The NoMIRACL dataset and the GPT-4 baseline are available at [project-miracl/nomiracl](https://github.com/project-miracl/nomiracl).

## Getting Started

To use the NoMIRACL dataset and replicate the experiments, follow the instructions in the [project-miracl/nomiracl](https://github.com/project-miracl/nomiracl) repository.

## Background and Problem Identification

Retrieval Augmented Generation (RAG) is essential for leveraging external knowledge in generating accurate responses. Large language models (LLMs) like GPT-3 and LLAMA-2 are widely used for RAG, but challenges exist in ensuring robust and reliable output.

The first-stage information retrieval system poses challenges in accurately retrieving relevant information, leading to errors in generated responses. No comprehensive evaluation of LLM reasoning capabilities in multiple languages has been conducted, leaving a gap in understanding robustness across different language resources.

## NoMIRACL Dataset Construction

NoMIRACL is a multilingual dataset designed to evaluate LLM robustness against errors in first-stage retrieval. The dataset covers 18 typologically diverse languages and includes two subsets: non-relevant and relevant.

### Non-Relevant Subset (F)
- Queries with no-known answers.
- All top-k passages manually judged as non-relevant.

### Relevant Subset (T)
- Queries with known answers.
- At least one of the top-k passages manually judged as relevant.

## Evaluation Metrics

We conduct a robustness evaluation using a binary classification task, comparing LLM predictions against the ground truth provided in NoMIRACL. The metrics used are hallucination rate and error rate.

- **Hallucination Rate:** Measures the model's tendency to hallucinate an answer when no answer is present in the non-relevant subset.

- **Error Rate:** Measures the model's inaccuracy in recognizing relevant passages in the relevant subset.

## NoMIRACL Dataset Overview

### Languages Covered:
Arabic, Bengali, German, English, Spanish, Persian, Finnish, French, Hindi, Indonesian, Japanese, Korean, Russian, Swahili, Thai, Yoruba, Chinese.

### Dataset Usage:
NoMIRACL is dynamic and customizable. Users can provide a sampling ratio for queries in the non-relevant to relevant subset, enabling flexibility in dataset utilization.

## Experimental Setup

### Baseline Model: GPT-4

We use GPT-4 as the baseline model for our experiments. GPT-4 is a closed-book LLM known for its multilingual capabilities.

#### Model Settings:
- Context window size: 4096 tokens
- Temperature score: 0.3
- Top-p sampling ratio: 0.95
- Maximum output length: 50 tokens

#### Vanilla Prompting:
We employ a zero-shot monolingual listwise prompting strategy using a vanilla prompt template. The template includes the input query and all top-k (oracle) passages, providing a short description of the task in English.

#### Cost Reduction:
To limit experimental costs, we truncate each passage to 390 tokens, and the evaluation is limited to a maximum of 250 randomly sampled queries for all languages in both NoMIRACL relevant and non-relevant splits.

## Conclusion

NoMIRACL offers a valuable dataset for assessing LLM robustness in RAG across diverse languages. The provided GPT-4 baseline highlights challenges and areas for improvement. We invite the community to explore and contribute to the advancement of LLM robustness in multilingual retrieval-augmented generation.

For more details, visit [project-miracl/nomiracl](https://github.com/project-miracl/nomiracl).

---

**Note:** This README provides an overview of the NoMIRACL project. For detailed information, instructions, and updates, refer to the official [project-miracl/nomiracl](https://github.com/project-miracl/nomiracl) repository.