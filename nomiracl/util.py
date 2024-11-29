from typing import List, Optional, Dict, Union
from tqdm.autonotebook import tqdm
from importlib.metadata import PackageNotFoundError, metadata

import os
import json
import shutil
import requests


# Referenced from https://github.com/UKPLab/sentence-transformers/blob/master/sentence_transformers/util.py
def check_package_availability(package_name: str, owner: str) -> bool:
    """
    Checks if a package is available from the correct owner.
    """
    try:
        meta = metadata(package_name)
        return meta["Name"] == package_name and owner in meta["Home-page"]
    except PackageNotFoundError:
        return False

def is_vllm_available() -> bool:
    """
    Returns True if the vllm-project `vllm` library is available.
    """
    return check_package_availability("vllm", "vllm-project")

def is_accelerate_available() -> bool:
    """
    Returns True if the huggingface accelerate library is available.
    """
    return check_package_availability("accelerate", "huggingface")

def is_bitsandbytes_available() -> bool:
    """
    Returns True if the bitsandbytes library is available.
    """
    return check_package_availability("bitsandbytes", "bitsandbytes-foundation")

def is_peft_available() -> bool:
    """
    Returns True if the peft library is available.
    """
    return check_package_availability("peft", "huggingface")


def count_word(sentence: str, word: str) -> int:
    # Split the sentence into words
    words = sentence.split()
    # Initialize a counter
    count = 0
    # Loop through the words and count occurrences of the specific word
    for w in words:
        if w == word:
            count += 1
    return count


def unzip_folder(folder_name):
    print("Unzipping " + folder_name)
    shutil.unpack_archive(folder_name + ".zip", folder_name)


def zip_folder(folder_name):
    print("Zipping " + folder_name)
    shutil.make_archive(folder_name, "zip", "../" + folder_name)


def chunks(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx : min(ndx + n, l)]


def download_url(url: str, save_path: str, chunk_size: int = 1024):
    """Download url with progress bar using tqdm
    https://stackoverflow.com/questions/15644964/python-progress-bar-and-downloads

    Args:
        url (str): downloadable url
        save_path (str): local path to save the downloaded file
        chunk_size (int, optional): chunking of files. Defaults to 1024.
    """
    r = requests.get(url, stream=True)
    total = int(r.headers.get("Content-Length", 0))
    with open(save_path, "wb") as fd, tqdm(
        desc=save_path,
        total=total,
        unit="iB",
        unit_scale=True,
        unit_divisor=chunk_size,
    ) as bar:
        for data in r.iter_content(chunk_size=chunk_size):
            size = fd.write(data)
            bar.update(size)


def save_results_as_jsonl(
    output_dir: str,
    results: Dict[str, Dict[str, Union[str, List[str]]]],
    qrels: Dict[str, Dict[str, int]],
    prompts: Dict[str, str],
    template: Optional[str] = None,
    filename: Optional[str] = "results.jsonl",
):
    """
    Save the results of generated output (results[model_name] ...) in JSONL format.

    Args:
        output_dir (str): The directory where the JSONL file will be saved.
        results (Dict[str, List[str]]): A dictionary containing the model results.
        prompts (List[str]): A list of prompts used for the model.
        query_ids (List[str]): A list of query IDs.
        template (Optional[str], optional): The template used for the model. Defaults to None.
        filename (Optional[str], optional): The name of the JSONL file. Defaults to 'results.jsonl'.
    """
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_dir), exist_ok=True)

    # Save results in JSONL format
    with open(os.path.join(output_dir, filename), "w") as f:
        json_output = {}
        for model_name in results:
            for query_id, model_output in results[model_name].items():
                if query_id not in json_output:
                    json_output[query_id] = {
                        "query_id": query_id,
                        "docids": [doc_id for doc_id in qrels[query_id]],
                        "prompt": prompts[query_id] if query_id in prompts else None,
                        "template": template if not None else "unknown",
                        "results": {model_name: model_output},
                    }
                else:
                    json_output[query_id]["results"][model_name] = model_output

        for query_id in json_output:
            f.write(json.dumps(json_output[query_id], ensure_ascii=False) + "\n")


def load_results_as_jsonl(input_filepath: str):
    results = {}
    with open(input_filepath, "r") as f:
        for line in f:
            json_line = json.loads(line)
            query_id = json_line["query_id"]
            for model_name, output in json_line["results"].items():
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][query_id] = output

    return results
