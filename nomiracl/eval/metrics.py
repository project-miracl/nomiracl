from typing import Dict, List, Union
from collections import Counter
from . import Metric

import logging
logger = logging.getLogger(__name__)

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
                 split: str,
                 results: Dict[str, Dict[str, Union[str, List[int]]]], **kwargs):
        
        if split not in ["valid", "invalid"]:
            raise ValueError("Invalid split name. Split name should be either 'valid' or 'invalid'")
        
        error_name = self.error_name[split]
        eval_results = {self.name: {}, error_name: {}, self.invalid_name: {}, "stats": {}}
        
        for model_name in results:
            eval_results[self.name][model_name] = 0.0
            eval_results["stats"][model_name] = {}

        for model_name in results:
            stats = dict(Counter(results[model_name].values()))
            eval_results["stats"][model_name] = stats

            if split == "invalid" and self.no_answer_str in stats:
                eval_results[self.name][model_name] = (stats[self.no_answer_str]) / sum(stats.values())
                if self.answer_str in stats: 
                    eval_results[error_name][model_name] = (stats[self.answer_str]) / sum(stats.values())
                else:
                    eval_results[error_name][model_name] = 0.0

            elif split == "valid" and self.answer_str in stats:
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
            