from typing import Optional

import json
import os
import pathlib

import logging
logger = logging.getLogger(__name__)

class Metric:
    def __init__(self, name, **kwargs):
        self.name = name

    def evaluate(self, qrels, results, **kwargs):
        raise NotImplementedError()
    
    @staticmethod
    def log_scores(scores: dict):
        """
        Log the evaluation scores.
        Args:
            scores (dict): A dictionary containing the evaluation scores.
        """
        for eval_result in scores:
            logger.info(f"{eval_result}:")
            
            if eval_result == "stats":
                for model_name in scores[eval_result]:
                    logger.info(f"\t{model_name}:")
                    for stat in scores[eval_result][model_name]:
                        logger.info(f"\t\t{stat}: {scores[eval_result][model_name][stat]}")
            else:
                for model_name in scores[eval_result]:
                    if type(scores[eval_result][model_name]) == float:
                        logger.info(f"\t{model_name}: {scores[eval_result][model_name]*100}%")
                
    
    @staticmethod
    def save_scores(scores: dict, filename: str, out_dir: Optional[str] = None):
        """
        Save the scores in JSON format.
        Args:
            scores (dict): A dictionary containing the evaluation scores.
            filename (str): The name of the JSON file.
            out_dir (str): The directory where the JSON file will be saved.
        """
        if out_dir is None:
            out_dir = os.path.join(pathlib.Path(__file__).parent.absolute(), "eval")
        
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, filename), "w") as f:
            f.write(json.dumps(scores, indent=4))
    
    @staticmethod
    def load_scores(filename: str):
        """
        Load the scores from a JSON file.
        Args:
            filename (str): The name of the JSON file.
        Returns:
            A dictionary containing the evaluation scores.
        """
        with open(filename, "r") as f:
            return json.load(f)