from nomiracl.util import count_word
from typing import List, Dict
import re


class PromptTemplate:
    """
    Args:
        `variables` (list[str]): a list of the variable names used in the template,
        `template` (str): The template string with {variable} names.
    """

    def __init__(self, 
                 variables: List[str] = None, 
                 template: str = None):
        self.variables = variables
        self.template = template

    def format(self, input_variables: Dict[str, str]) -> str:
        """
        Returns the prompt using the `input_variables` in the form of {"query": "text", ...} to a string
        """
        return self.template.format(**input_variables)

    def get_template(self) -> str:
        """
        Returns the template string.
        """
        return self.template


class VanillaTemplate(PromptTemplate):
    def __init__(self, count: int = 10):        
        self.max_count = count
        self.no_answer = "I don't know"
        self.answer = "Yes, answer is present"
        self.invalid_answer = "Invalid"
        self.passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + self.passage_variables
        self.template = ("I will give you a question and several contexts containing information about the question." +
        f" Read the contexts carefully. If any of the contexts answers the question, respond as either \"{self.answer}\" or \"{self.no_answer}\"." +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(self.passage_variables, 1)]) + 
        "\n\nOUTPUT:\n")

    def __call__(self, query: str, passages: List[str]) -> str:
        
        if len(passages) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        variables = {"query": query}
        for idx, passage in enumerate(passages):
            variables["passage_{}".format(idx + 1)] = passage
        
        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response: str) -> str:
        """
        Postprocesses the model output to extract the answer.
        """
        # postprocess the response (lower case, remove newlines and tabs)
        response = response.replace("\n", " ").replace("\t", " ").lower()
        regex = re.findall(f"\[(\d+|{self.max_count})\]", response)
        no_answer = self.no_answer.lower()
        answer = self.answer.lower()
        
        # Responses where references are present in model output.
        if no_answer in response and answer in response:
            if count_word(response, answer) > 1:
                return self.answer
            elif count_word(response, no_answer) == 1 and count_word(response, answer) == 1:
                return self.invalid_answer
            else:
                return self.invalid_answer
        
        elif no_answer in response:
            return self.no_answer
        
        elif answer in response:
            return self.answer
        
        # edge cases
        else:
            if "answer is present" in response:
                return self.answer
            elif "answer is not present" in response:
                return self.no_answer
            elif len(regex) and len(response) < 10:
                return self.answer
            else:   
                return self.invalid_answer

# Ablations of the Vanilla Template
class RoleTemplate(VanillaTemplate):
    def __init__(self, count: int = 10):
        super().__init__(count)
        self.template = ("You are an evaluator checking whether the question contains the answer within the contexts or not. I will give you a question and several contexts containing information about the question." +
        f" Read the contexts carefully. If any of the contexts answers the question, respond as either \"{self.answer}\" or \"{self.no_answer}\". Do not add any other information in your output." +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(self.passage_variables, 1)]) + 
        "\n\nOUTPUT:\n")

class RepeatTemplate(VanillaTemplate):
    def __init__(self, count: int = 10):
        super().__init__(count)
        self.template = ("I will give you a question and several contexts containing information about the question." +
        f" Read the contexts carefully. If any of the contexts answers the question, respond as either \"{self.answer}\" or \"{self.no_answer}\"." +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(self.passage_variables, 1)]) + 
        "\n\nRemember to read the contexts carefully. If any of the contexts answers the question: {query}, "
        f"respond as either \"{self.answer}\" or \"{self.no_answer}\"." +
        "\n\nOUTPUT:\n")

class ExplanationTemplate(VanillaTemplate):
    def __init__(self, count: int = 10):
        super().__init__(count)
        self.template = ("I will give you a question and several contexts containing information about the question." +
        f" Read the contexts carefully and provide a step-by-step explanation for your answer. If any of the contexts answers the question, respond as either \"{self.answer}\" or \"{self.no_answer}\"." +  
        f" You must follow the output format with: ## Explanation:... ## Answer: \"{self.answer}\" OR \"{self.no_answer}\"" +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(self.passage_variables, 1)]) + "\n\n")
