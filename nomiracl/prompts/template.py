from typing import List, Dict
import re
from .exemplar import FewShotExemplars, OrcaDPOExemplars

LANG_ISO_MAP = {
    "en": "English",
    "bn": "Bengali",
    "zh": "Chinese",
    "fr": "French",
    "de": "German",
    "ja": "Japanese",
    "ru": "Russian",
    "es": "Spanish",
    "sw": "Swahili",
    "th": "Thai",
}

def count_word(sentence, word):
    # Split the sentence into words
    words = sentence.split()
    # Initialize a counter
    count = 0
    # Loop through the words and count occurrences of the specific word
    for w in words:
        if w == word:
            count += 1
    return count

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


class YESNOTemplate(PromptTemplate):
    def __init__(self, count=10):        
        self.max_count = count
        self.no_answer = "I don't know"
        self.answer = "Yes, answer is present"
        self.invalid_answer = "Invalid"
        passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + passage_variables
        self.template = ("I will give you a question and several contexts containing information about the question." +
        f" Read the contexts carefully. If any of the contexts answers the question, respond as either \"{self.answer}\" or \"{self.no_answer}\"." +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(passage_variables, 1)]) + 
        "\n\nOUTPUT:\n")

    def __call__(self, query, passages):
        
        if len(passages) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        variables = {"query": query}
        for idx, passage in enumerate(passages):
            variables["passage_{}".format(idx + 1)] = passage
        
        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response: str):
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

class FewShotPointwiseTemplate(PromptTemplate):
    def __init__(self, count=1):
        self.max_count = count
        self.invalid_answer = "Invalid"
        self.no_answer = "I don't know"
        self.answer = "Yes, answer is present"
        self.variables = ["query", "passage"]
        self.exemplar = FewShotExemplars()

        self.template = ("You are in a Job interview. I will provide you a question and a context containing information about the question." +
                         f" Read the given context carefully and respond whether the context answers the question. Answer either \"{self.answer}\" or \"{self.no_answer}\"."
                         "\n\nQUESTION: {query_1}\nCONTEXT: {passage_1}\nOUTPUT: {answer_1}"
                         "\n\nQUESTION: {query_2}\nCONTEXT: {passage_2}\nOUTPUT: {answer_2}"
                         "\n\nQUESTION: {query_3}\nCONTEXT: {passage_3}\nOUTPUT: {answer_3}"
                         "\n\nQUESTION: {query_4}\nCONTEXT: {passage_4}\nOUTPUT: {answer_4}"
                         "\n\nQUESTION: {query}\nCONTEXT: {passage}\nOUTPUT:")

    def __call__(self, query, passage, **kwargs):
        
        if len([passage]) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        examples = self.exemplar.get_exemplars(**kwargs)
        variables = {"query": query, "passage": passage}
        
        for idx, ex in enumerate(examples):
            # positive passage first
            
            idy = (idx * 2) + 1
            pos_passage = ex["positive"]["title"] + ": " + ex["positive"]["text"]
            variables["query_{}".format(idy)] = ex["query"]
            variables["passage_{}".format(idy)] = pos_passage
            variables["answer_{}".format(idy)] = self.answer
            idy = idy + 1
            neg_passage = ex["negative"]["title"] + ": " + ex["negative"]["text"]
            variables["query_{}".format(idy)] = ex["query"]
            variables["passage_{}".format(idy)] = neg_passage
            variables["answer_{}".format(idy)] = self.no_answer

        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response_list: List[str]):
        """
        Postprocesses the model output to extract the answer.
        """
        output_response = []
        for response in response_list:
            # postprocess the response (lower case, remove newlines and tabs)
            if "\n\n" in response:
                response = response.split("\n\n")[0]
            elif "\n" in response:
                response = response.split("\n")[0]
            
            response = response.replace("\n", " ").replace("\t", " ").lower()
            no_answer = self.no_answer.lower()
            answer = self.answer.lower()

            if no_answer in response and answer in response:
                output_response.append(self.invalid_answer)
            elif no_answer in response:
                output_response.append(self.no_answer)
            elif answer in response:
                output_response.append(self.answer)
            else:
                output_response.append(self.invalid_answer)
        
        return output_response

class FewShotRelevanceTemplate(PromptTemplate):
    def __init__(self, count=1):
        self.max_count = count
        self.invalid = "Invalid"
        self.rel = [0,1,2]
        self.variables = ["query", "passage"]
        self.exemplar = FewShotExemplars()
        self.exemplar_rel = {1: 2, 2: 0, 3: 2, 4: 1}

        self.template = (f"From a scale of {min(self.rel)} to {max(self.rel)}, judge the relevance between the query and document. " +
                         "Do not mention anything else in your preamble." +
                         "\n\Query: {query_1}\n" + "Document: {passage_1}\nOutput: {answer_1}"
                         "\n\nQuery: {query_2}\n" + "Document: {passage_2}\nOutput: {answer_2}"
                         "\n\nQuery: {query_3}\n" + "Document: {passage_3}\nOutput: {answer_3}"
                         "\n\nQuery: {query_4}\n" + "Document: {passage_4}\nOutput: {answer_4}"
                         "\n\nQuery: {query}\n" + "Document: {passage}\nOutput:")

    def __call__(self, query, passage, **kwargs):
        
        if len([passage]) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        examples = self.exemplar.get_exemplars(**kwargs)
        variables = {"query": query, "passage": passage}
        
        for idx, ex in enumerate(examples):
            # positive passage first
            
            idy = (idx * 2) + 1

            pos_passage = ex["positive"]["title"] + ": " + ex["positive"]["text"]
            variables["query_{}".format(idy)] = ex["query"]
            variables["passage_{}".format(idy)] = pos_passage
            variables["answer_{}".format(idy)] = self.exemplar_rel[idy]
            idy = idy + 1
            neg_passage = ex["negative"]["title"] + ": " + ex["negative"]["text"]
            variables["query_{}".format(idy)] = ex["query"]
            variables["passage_{}".format(idy)] = neg_passage
            variables["answer_{}".format(idy)] = self.exemplar_rel[idy]

        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response_list: List[str]):
        """
        Postprocesses the model output to extract the answer.
        """
        # postprocess the response (lower case, remove newlines and tabs)
        output_list = []
        for response in response_list:
            response = response.replace("\n", " ").replace("\t", " ").lower()
            regex = re.findall(f"(\d+|{max(self.rel)})", response)
            if len(regex) > 0:
                output_list.append(int(regex[0]))
            else:
                output_list.append(self.invalid)
                
        return output_list
            

class HAGRIDTemplate(PromptTemplate):
    def __init__(self, count=10):
        self.max_count = count
        self.no_answer = "I don't know"
        passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + passage_variables
        self.template = ("I will give you a question and several contexts containing information about the question." +
        " Read the given contexts carefully and check if any contexts answer the question. " +
        " If any of the given contexts answer the question, respond as . " +
        # " Also, mention the reference of parts of your answer based on the given contexts" +
        # " within brackets [] as in the standard IEEE format.\n\n"
        f" Alternatively, If none of the given contexts answer the question, respond as \"{self.no_answer}\"." +
        " Do not make up any references in your answer outside the given contexts. " +
        f"Finally, provide an valid explanation for your answer." +
        # " Also, mention the reference of parts of your answer based on the given contexts" +
        # " within brackets [] as in the standard IEEE format." + "\n\n" +
        # " Do not make up any references in your answer that are not present in the given contexts. " +
        # f"If none of the contexts provide an answer the question, write '{self.no_answer}' in the answer section." +
        "\n\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n".join(["[{}] {}".format(i, "{" + passage + "}") for i, passage in enumerate(passage_variables, 1)]) + 
        "\n\nANSWER:")

    def __call__(self, query, passages):
        
        if len(passages) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        variables = {"query": query}
        for idx, passage in enumerate(passages):
            variables["passage_{}".format(idx + 1)] = passage
        
        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response: str):
        """
        Postprocesses the model output to extract the answer.
        """
        # postprocess the response (lower case, remove newlines and tabs)
        response = response.replace("\n", " ").replace("\t", " ").lower()
        regex = re.findall(f"\[(\d+|{self.max_count})\]", response)
        
        # LLAMA-based responses where references are present in model output.
        if "references:" in response:
            response_prefix = response.split("references:")[0]
            regex_prefix = re.findall(f"\[(\d+|{self.max_count})\]", response_prefix)
            answers = [int(x) for x in sorted(set(regex_prefix))]
            
            # Hallucinated answers, where the answer is not present in the context.
            if len(answers) == self.max_count: return self.no_answer
            else:
                response_postfix = response.split("references:")[1]
                regex_postfix = re.findall(f"\[(\d+|{self.max_count})\]", response_postfix)
                answers = [int(x) for x in sorted(set(regex_postfix))]
                # if all 10 references are present in the context, return no answer.
                if len(answers) == self.max_count: return self.no_answer
                # if the number of references in the context is zerp, return no answer.
                elif len(answers) == 0: return self.no_answer
                # else return the answers.
                else: return answers
        
        # If "No Answer" is present in the response, return "No Answer".
        if self.no_answer.lower() in response:
            return self.no_answer
        
        # Else if the regex is greater than 0, check if answers less than 10.
        elif len(regex) > 0:
            answers = [int(x) for x in sorted(set(regex))]
            if len(answers) < self.max_count:
                return answers
            elif len(answers) == self.max_count:
                return self.no_answer
        else:
            return self.no_answer

class RerankTemplate(PromptTemplate):
    def __init__(self, count=10):
        self.max_count = count
        self.no_answer = "No Answer"
        passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        score_variables = [f"score_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + passage_variables + score_variables
        self.template = (f"Read the following question and contexts with scores and choose the correct contexts to answer the question." +
                         f"Scores are ranging from '-1' to '1'. A context with higher score should have higher probability to be cited in your answer." +
                         f"If none of the contexts answer the question, write '{self.no_answer}' in the answer section." +
        "\nQUESTION:\n{query}\n\n" + "CONTEXTS:\n" + 
        "\n".join(["[{}] score: ({}) {}".format(i, "{" + score_variables[i-1] + "}", "{" + passage + "}") for i, passage in enumerate(passage_variables, 1)]) + 
        "\n\n" + "ANSWER:")

    def __call__(self, query, passages, scores):
        
        if len(passages) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        variables = {"query": query}
        for idx, passage in enumerate(passages):
            variables["passage_{}".format(idx + 1)] = passage
            variables["score_{}".format(idx + 1)] = "{:.2f}".format(scores[idx])
        
        prompt = self.format(variables)
        return prompt
    
class InstructQATemplate(PromptTemplate):
    def __init__(self, count=10):
        self.max_count = count
        self.no_answer = "I don't know"
        passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + passage_variables
        self.template = ("Please answer the following question given the following passages. " +
                         f"If the answer is not in the passages or cannot be inferred from the passages, respond as \"{self.no_answer}\".\n" +
                        "\n\n".join(["- title: {}".format("{" + passage + "}") for i, passage in enumerate(passage_variables, 1)]) + 
                        "\n\n" + "Question: {query}\nAnswer:")

    def __call__(self, query, passages):
        
        if len(passages) != self.max_count:
            raise ValueError("Number of passages should be equal to {}".format(self.max_count))
        
        variables = {"query": query}
        for idx, passage in enumerate(passages):
            variables["passage_{}".format(idx + 1)] = passage
        
        prompt = self.format(variables)
        return prompt
    
    def postprocess(self, response: str):
        """
        Postprocesses the model output to extract the answer.
        """
        response = response.replace("\n", " ").replace("\t", " ").lower()
        if self.no_answer.lower() in response:
            return "No Answer"
        else:
            return "Yes. Answer is present."
    
class OrcaDPOTranslationTemplate(PromptTemplate):
    def __init__(self, count=10):
        self.max_count = count
        self.no_answer = "I don't know"
        self.exemplar = OrcaDPOExemplars()
        passage_variables = [f"passage_{i}" for i in range(1, self.max_count + 1)]
        self.variables = ["query"] + passage_variables
        self.template = ("<s> [INST] You are a professional {lang} translator and spelling corrector. Please translate the given question and the chosen and rejected answer into {lang}.\nBelow is an example: [/INST] \n\n" +
                         "Q0: Please answer the following question: Moeenuddin Ahmad Qureshi - Moeenuddin Ahmad Qureshi usually referred to as Moeen Qureshi (born 1930) is a Pakistani economist and political figure. A former Vice President of the World Bank he was the Interim Prime Minister of Pakistan from July 18 1993 until 19 October 1993. Given a choice of categories company, educational institution, artist, athlete, office holder, mean of transportation, building, natural place, village, animal, plant, album, film or written work, the text refers to which one? Answer:\n" +
                         "C0: The text refers to the category of \"office holder\" as Moeenuddin Ahmad Qureshi served as the Interim Prime Minister of Pakistan from July 18, 1993, until October 19, 1993.\n" +
                         "R0: Sure, I'd be happy to help! Based on the information provided, the text refers to the category of \"office holder\" because Moeenuddin Ahmad Qureshi is a former Interim Prime Minister of Pakistan.\n\n" +
                         "T-Q0: {ex_question}\nT-C0: {ex_chosen}\nT-R0: {ex_rejected} </s> \n\n" +
                         "[INST] Please keep in mind that: (1) Keep the translations consistent for names of people and places within the sentences.\n" +
                         "(2) You must translate the text into {lang}.\n" +
                         "(3) You must follow the output format with: \"T-Q1:... T-C1:... T-R1:...\"" +
                         "\n\nQ1: {question}\nC1: {chosen}\nR1: {rejected}\n\ntranslation: [/INST]")
        self.language_dict = {

        }

    def __call__(self, question, chosen, rejected, language):
        examples = self.exemplar.get_exemplars(language=language)
        variables = {"question": question.replace("\n", " "), "chosen": chosen.replace("\n", " "), "rejected": rejected.replace("\n", " "), "lang": LANG_ISO_MAP[language],
                     "ex_question": examples["prompt"], "ex_chosen": examples["chosen"], "ex_rejected": examples["rejected"]}
        prompt = self.format(variables)
        return prompt

    def postprocess(self, response: str):
        """
        Postprocesses the model output to extract the answer.
        """
        question, chosen, rejected = "", "", ""
        if "T-Q1" in response and "T-C1" in response:
            question = response.split("T-Q1:")[1].split("T-C1:")[0].strip()
        if "T-C1" in response and "T-R1" in response:
            chosen = response.split("T-C1:")[1].split("T-R1:")[0].strip()
            rejected = response.split("T-R1:")[1].strip()
        return {"question": question, "chosen": chosen, "rejected": rejected}
