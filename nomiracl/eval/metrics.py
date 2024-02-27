from typing import Dict, List, Union, Optional
from collections import Counter

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


class ConfusionMetric(Metric):
    def __init__(self, 
                 name='Accuracy',
                 error_name={"relevant": "Error", "non_relevant": "Hallucination"},
                 invalid_name="Invalid",
                 no_answer_str="I don't know",
                 answer_str="Yes, answer is present"):
        self.no_answer_str = no_answer_str
        self.answer_str = answer_str
        self.error_name = error_name
        self.invalid_name = invalid_name
        super().__init__(name=name)
        
    def evaluate(self, 
                 subset: str,
                 results: Dict[str, Dict[str, Union[str, List[int]]]], **kwargs) -> Dict[str, Dict[str, float]]:
        """ Evaluate the results for the given subset."""
        
        if subset not in ["relevant", "non_relevant"]:
            raise ValueError("Split name incorrect. Split name should be either 'relevant' or 'non_relevant'")
        
        error_name = self.error_name[subset]
        eval_results = {self.name: {}, error_name: {}, self.invalid_name: {}, "stats": {}}
        
        for model_name in results:
            eval_results[self.name][model_name] = 0.0
            eval_results["stats"][model_name] = {}

        for model_name in results:
            stats = dict(Counter(results[model_name].values()))
            eval_results["stats"][model_name] = stats

            if subset == "non_relevant" and self.no_answer_str in stats:
                eval_results[self.name][model_name] = (stats[self.no_answer_str]) / sum(stats.values())
                if self.answer_str in stats: 
                    eval_results[error_name][model_name] = (stats[self.answer_str]) / sum(stats.values())
                else:
                    eval_results[error_name][model_name] = 0.0

            elif subset == "relevant" and self.answer_str in stats:
                eval_results[self.name][model_name] = (stats[self.answer_str]) / sum(stats.values())
                if self.no_answer_str in stats: 
                    eval_results[error_name][model_name] = (stats[self.no_answer_str]) / sum(stats.values())
                else:
                    eval_results[error_name][model_name] = 0.0
            
            if self.invalid_name in stats:
                eval_results[self.invalid_name][model_name] = (stats[self.invalid_name]) / sum(stats.values())
        
        # log the evaluation scores
        self.log_scores(eval_results)
        
        return eval_results
            