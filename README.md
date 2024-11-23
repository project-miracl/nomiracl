# NoMIRACL: A Multilingual Relevance Assessment Dataset for RAG Applications
<p align="center">
    <a href="https://aclanthology.org/2024.findings-emnlp.730/">
        <img alt="EMNLP2024" src="https://img.shields.io/badge/Citation-EMNLP_2024-orange.svg">
    </a>
    <a href="https://github.com/project-miracl/nomiracl">
        <img alt="Stars" src="https://img.shields.io/github/stars/project-miracl/nomiracl.svg?style=flat&logo=github&colorB=blue&label=stars">
    </a>
    <a href="https://www.python.org/">
            <img alt="Build" src="https://img.shields.io/badge/Made%20with-Python-1f425f.svg?color=purple">
    </a>
    <a href="https://github.com/project-miracl/nomiracl/blob/main/LICENSE">
        <img alt="License" src="https://img.shields.io/github/license/project-miracl/nomiracl.svg?style=flat&colorB=green">
    </a>
</p>

<h4 align="center">
    <a href="./"><img style="float: middle;" width="800" height="570" src="./images/nomiracl-teaser.png" /></a>
    <footer><br clear="all"/>The image has been generated using miramuseai.net and Adobe photoshop.</footer>
</h4>

NoMIRACL [[EMNLP'24 Findings]](https://aclanthology.org/2024.findings-emnlp.730/) is a multilingual relevance assessment dataset for evaluating query \& passage relevancy in large language models (LLMs). This is extremely useful in RAG settings, i.e., when a retrieval systems retrieves a subset of passages or documents which either can or cannot be relevant to the user query. The LLM (as the generator) should assess the relevancy and only answer -- *if* a relevant passage is found within the subset, else abstain from answering.

**This repository provides starter code to evaluate diverse multilingual LLMs using our prompt template on NoMIRACL.**

For more information, checkout out our publication:
- [“Knowing When You Don’t Know”: A Multilingual Relevance Assessment Dataset for Robust Retrieval-Augmented Generation](https://aclanthology.org/2024.findings-emnlp.730/) (Thakur et al., :star: EMNLP 2024 Findings)


## :wrench: Installation
You can install NoMIRACL code repository via pip:

```python
pip install nomiracl
```

If you want to build from source, use:

```bash
$ git clone https://github.com/project-miracl/nomiracl.git
$ cd nomiracl
$ pip install -e .
```

## :star: Getting Started

#### 1. Loading NoMIRACL Dataset 
- 50\% of relevant examples, 50\% of non-relevant, both maximum capped at 250. 
- Full example available in [sample_load_no_miracl.py](./examples/sample_load_no_miracl.py).
```python
from nomiracl.dataset import NoMIRACLDataLoader

data_loader = NoMIRACLDataLoader(language = "english", 
                                 split = "test", # or 'dev' 
                                 hf_dataset_name="miracl/nomiracl", 
                                 load_from_huggingface=True)
                        
corpus, queries, qrels = data_loader.load_data_sample(
    relevant_ratio = 0.5, non_relevant_ratio = 0.5, max_sample_pool = 250)
```

#### 2. LLM prompt generation
- Full example available in [sample_model_generation.py](./examples/sample_model_generation.py).
```python
from nomiracl.generation.utils import load_model

model_name = "meta-llama/Meta-Llama-3-8B-Instruct"

# List of techniques supported in nomiracl: 
# huggingface (GPUs), vllm (GPUs), cohere (API), openai (API), nvidia (API), azure (API), anyscale (API)
# `cohere` requires COHERE_API_KEY, `openai` requires OPENAI_API_KEY, `nvidia` requires NVIDIA_API_KEY
# `azure` requires AZURE_OPENAI_API_BASE, AZURE_OPENAI_API_VERSION and AZURE_OPENAI_API_KEY
# `anyscale` requires ANYSCALE_BASE_URL and ANYSCALE_API_KEY.

technique = "vllm" # or huggingface or nvidia, anyscale etc.

model = load_model(
    technique, # technique
    model_name, # model_name
    cache_dir="<your-cache-dir>", # extra kwargs
    batch_size=2, # extra kwargs
    num_gpus=1, # extra kwargs
    concurrency=2 # extra kwargs
)

# Sample prompts
prompts = [
    "What is the capital of France?",
    "What is the capital of Germany?",
    "What is the capital of Italy?",
]

model_results = model.call(prompts)

for prompt, result in zip(prompts, model_results):
    print("Prompt: {}".format(prompt))
    print("{} result: {}".format(model_name, result))
```

#### 3. Loading our paper used prompt templates
- Full example available in [sample_vanilla_prompt_exploration.py](./examples/sample_vanilla_prompt_exploration.py).

```python
from nomiracl.prompts.utils import load_prompt_template

# Options include: vanilla, role, repeat, explanation 
prompt_cls = load_prompt_template("vanilla", count = 10) # as we include 10 passages

query = "Which is the best programming language?"

passages = [
    "Python is the best programming language.",
    "Javascript is the best programming language.",
    "Go is the best programming language.",
    "Java is the best programming language.",
    "C# is the best programming language.",
    "Ruby is the best programming language.",
    "R is the best programming language.",
    "C++ is the best programming language.",
    "C is the best programming language.",
    "Rust is the best programming language.",
]

prompt = prompt_cls(query=query, passages=passages)
```

Or you can provide your **own** custom prompt template by modifying the `self.template` in `nomiracl.VanillaTemplate`.

```python
from nomiracl.prompts import VanillaTemplate

class CustomTemplate(VanillaTemplate):
    def __init__(self, count: int = 1):
        super().__init__(count)
        self.template = (
            "This is a pairwise prompt template. Respond as either "{self.answer}" or "{self.no_answer}".'
            + "\n\nQUESTION:\n{query}\n\n"
            + "CONTEXT:\n"
            + "\n\n".join(
                [
                    "[{}] {}".format(i, "{" + passage + "}")
                    for i, passage in enumerate(self.passage_variables, 1)
                ]
            )
            + "\n\nOUTPUT:\n"
        )
```

## :hugs: NoMIRACL Dataset

The NoMIRACL dataset is available in HuggingFace under: `miracl/nomiracl`.

Languages covered: Arabic (ar), Bengali (bn), German (de), English (en), Spanish (es), Persian (fa), Finnish (fi), French (fr), Hindi (hi), Indonesian (id), Japanese (ja), Korean (ko), Russian (ru), Swahili (sw), Thai (th), Yoruba (yo), Chinese (zh).

HuggingFace Page: [https://huggingface.co/datasets/miracl/nomiracl](https://huggingface.co/datasets/miracl/nomiracl) 

```python
import datasets

language = 'german'  # or any of the 18 languages
subset = 'relevant'  # or 'non_relevant'
split = 'test'       # or 'dev' for development split

# four combinations available: 'dev.relevant', 'dev.non_relevant', 'test.relevant' and 'test.non_relevant'
nomiracl = datasets.load_dataset('miracl/nomiracl', language, split=f'{split}.{subset}')
```

### Baseline Accuracy on NoMIRACL non-relevant subset (test split, maximum cap of 250 per language)

Baseline results (250 queries) are available within the repository under `./results/baselines/non_relevant`.

An example datapoint under `./results/baselines/non_relevant/en.test.vanilla_prompt.jsonl`
```
{
    "query_id": "842558#0", 
    "docids": ["2842207#5", "7004944#45", "3310762#14", "47220460#1", "36451733#7", "3310762#20", "4724576#4", "22373402#0", "52203230#0", "23126218#4"], 
    "prompt": "I will give you a question and several contexts containing information about the question. [ ... ] \n\nOUTPUT:\n", 
    "template": "vanilla", 
    "results": {"gpt-4-azure": "Yes, answer is present.", 
                "llama-2-13b-chat": "\nYes, answer is present in [6].\n\nNo answers found in the other contexts.",
                [...]
                "aya-101": "Wales"}
}
```

### Baseline Accuracy on NoMIRACL relevant subset (test split, maximum cap of 250 per language)

Baseline results (250 queries) are available within the repository under `./results/baselines/relevant`.

An example datapoint under `./results/baselines/relevant/en.test.vanilla_prompt.jsonl`
```
{
    "query_id": "8706103#0", 
    "docids": ["42057469#2", "4998067#1", "29247933#0", "162619#81", "422315#13", "26790310#4", "41298602#18", "22816#16", "123427#61", "23576525#0"], 
    "prompt": "I will give you a question and several contexts containing information about the question. [ ... ] \n\nQUESTION:\nWhat is the course that will be discontinued as defined by the National Education Policy? [ ... ] \n\nOUTPUT:\n", 
    "template": "vanilla", 
    "results": {"gpt-4-azure": "I don't know.", 
                "llama-2-13b-chat": "Please answer the question based on the given contexts.",
                [...]
                "aya-101": "I don't know"}
}
```

## NoMIRACL Dataset Construction

<img src="./images/NoMIRACL-Flowchart.drawio.png" width="1013" height="179" />

NoMIRACL is a multilingual dataset designed to evaluate LLM robustness in relevance assessment to help avoid errors in first-stage retrieval. The dataset covers 18 typologically diverse languages and includes two subsets: non-relevant and relevant.

### Non-Relevant Subset (F)
- Queries with no-known answers within the retrieved oracle passages.
- All top-k passages manually judged as non-relevant (relevancy score = 0).

### Relevant Subset (T)
- Queries with known answers within the retrieved oracle passages.
- At least one of the top-k passages manually judged as relevant (relevancy score = 1).

## Evaluation Metrics

<img src="./images/NoMIRACL-confusion-matrix.png" width="411" height="193"/>

We conduct a robustness evaluation using a binary classification task, comparing LLM predictions against the ground truth provided in NoMIRACL. The metrics used are hallucination rate and error rate.

- **Hallucination Rate:** `FP/(FP + TN)` Measures the model's tendency to hallucinate an answer when no answer is present in the non-relevant subset.

- **Error Rate:** `FN/(FN + TP)` Measures the model's inaccuracy in recognizing relevant passages in the relevant subset.

## :handshake: Collaboration and Acknowledgements

The NoMIRACL dataset has been made possible due to a collaborative effort of the following universities and organizations:

- University of Waterloo
- Huawei Noah's Ark Lab

Parts of the NoMIRACL code structure has been inspired by:
- [https://github.com/McGill-NLP/instruct-qa](https://github.com/McGill-NLP/instruct-qa)

## :scroll: Citations

If you use NoMIRACL or parts in a research paper, please cite our work as follows:

```
@article{thakur:2024,
  author       = {Nandan Thakur and
                  Luiz Bonifacio and
                  Xinyu Zhang and
                  Odunayo Ogundepo and
                  Ehsan Kamalloo and
                  David Alfonso{-}Hermelo and
                  Xiaoguang Li and
                  Qun Liu and
                  Boxing Chen and
                  Mehdi Rezagholizadeh and
                  Jimmy Lin},
  title        = {NoMIRACL: Knowing When You Don't Know for Robust Multilingual Retrieval-Augmented
                  Generation},
  journal      = {CoRR},
  volume       = {abs/2312.11361},
  year         = {2023},
  url          = {https://doi.org/10.48550/arXiv.2312.11361},
  doi          = {10.48550/ARXIV.2312.11361},
  eprinttype    = {arXiv},
  eprint       = {2312.11361},
  timestamp    = {Tue, 16 Jan 2024 11:57:42 +0100},
  biburl       = {https://dblp.org/rec/journals/corr/abs-2312-11361.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}

```

---
Contact person: Nandan Thakur, [nandan.thakur@uwaterloo.co](mailto:nandan.thakur@uwaterloo.ca)

> This repository contains experimental software and is published for the sole purpose of giving additional background details on the respective publication.
