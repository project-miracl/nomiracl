from typing import List, Optional, Dict, Tuple, Union

import os
import json
import shutil
from pprint import pprint

def unzip_folder(folder_name):
    print("Unzipping " + folder_name)
    shutil.unpack_archive(folder_name + '.zip', folder_name)


def zip_folder(folder_name):
    print("Zipping " + folder_name)
    shutil.make_archive(folder_name, 'zip', "../" + folder_name)


def save_results_as_jsonl(output_dir: str,
                         results: Dict[str, Dict[str, Union[str, List[str]]]],
                         qrels: Dict[str, Dict[str, int]],
                         prompts: Dict[str, str],
                         template: Optional[str] = None,
                         filename: Optional[str] = 'results.jsonl', 
                         postprocess_results: Optional[Dict[str, Dict[str, Union[str, List[str]]]]] = {}):
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
    with open(os.path.join(output_dir, filename), 'w') as f:
        json_output = {}
        for model_name in results:
            for query_id, model_output in results[model_name].items():
                if query_id not in json_output:
                    json_output[query_id] = {
                        'query_id': query_id, 
                        'docids': [doc_id for doc_id in qrels[query_id]],
                        'prompt': prompts[query_id] if query_id in prompts else None,
                        'template': template if not None else 'unknown',
                        'results': {model_name: model_output}
                        }
                else:
                    json_output[query_id]['results'][model_name] = model_output
        
        # Postprocess results
        if len(postprocess_results):
            for model_name in postprocess_results:
                for query_id, output in postprocess_results[model_name].items():
                    if "postprocess_results" not in json_output[query_id]:
                        json_output[query_id]['postprocess_results'] = {model_name: output}
                    else:
                        json_output[query_id]['postprocess_results'][model_name] = output
        
        for query_id in json_output:
            f.write(json.dumps(json_output[query_id], ensure_ascii=False) + '\n')

def load_results_as_jsonl(input_filepath: str):
    results, postproc_results = {}, {}
    with open(input_filepath, 'r') as f:
        for line in f:
            json_line = json.loads(line)
            query_id = json_line['query_id']
            for model_name, output in json_line['results'].items():
                if model_name not in results:
                    results[model_name] = {}
                results[model_name][query_id] = output
            
            if "postprocess_results" in json_line:
                for model_name, output in json_line['postprocess_results'].items():
                    if model_name not in postproc_results:
                        postproc_results[model_name] = {}
                    postproc_results[model_name][query_id] = output
            
    return results, postproc_results
