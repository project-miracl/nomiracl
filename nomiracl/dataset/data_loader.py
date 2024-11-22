from .util import load_corpus, load_queries, load_qrels

from typing import Dict, Tuple
import os
import logging
import random
import datasets

logger = logging.getLogger(__name__)


class DataLoader:
    def __init__(
        self,
        corpus_file: str = "corpus.jsonl",
        query_file: str = "topics.tsv",
        qrels_file: str = "qrels.tsv",
        **kwargs,
    ):
        self.corpus_file = corpus_file
        self.query_file = query_file
        self.qrels_file = qrels_file

        self.corpus = {}
        self.queries = {}
        self.qrels = {}

    @staticmethod
    def check(fIn: str, ext: str):
        if not os.path.exists(fIn):
            raise ValueError(
                "File {} not present! Please provide accurate file.".format(fIn)
            )

        if not fIn.endswith(ext):
            raise ValueError(
                "File {} must be present with extension {}".format(fIn, ext)
            )

    def load_data(
        self,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_file, ext="tsv")
        self.check(fIn=self.qrels_file, ext="tsv")

        if not len(self.corpus):
            logger.info("Loading Corpus...")
            self.corpus = load_corpus(filepath=self.corpus_file)
            logger.info("Loaded %d Documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries):
            logger.info("Loading Queries...")
            self.queries = load_queries(filepath=self.query_file)

        if os.path.exists(self.qrels_file):
            self.qrels = load_qrels(filepath=self.qrels_file)
            self.queries = {
                qid: self.queries[qid] for qid in self.qrels if qid in self.queries
            }
            logger.info("Loaded %d Queries.", len(self.queries))
            logger.info("Query Example: %s", list(self.queries.values())[0])

        return self.corpus, self.queries, self.qrels


class NoMIRACLDataLoader(DataLoader):

    def __init__(
        self,
        data_dir: str = None,
        split: str = "test",
        language: str = "en",
        corpus_file: str = "corpus.jsonl",
        query_files: Dict[str, str] = None,
        qrels_files: Dict[str, str] = None,
        relevant: Dict[bool, str] = {True: "relevant", False: "non_relevant"},
        load_from_huggingface: bool = False,
        hf_dataset_name: str = "miracl/nomiracl",
    ):
        """
        Load the NoMIRACL dataset from the given directory.
        Args:
            data_dir (str): The directory where the dataset is stored.
            split (str): The split of the dataset to be used. Default is "test".
            corpus_file (str): The name of the corpus file. Default is "corpus.jsonl".
            query_files (Dict[str, str]): The dictionary containing the relevant and non-relevant query files. Default is None.
            qrels_files (Dict[str, str]): The dictionary containing the relevant and non-relevant qrels files. Default is None.
            relevant (Dict[bool, str]): The dictionary containing the relevant and non-relevant strings. Default is {True: "relevant", False: "non_relevant"}.
            load_from_huggingface (bool): Whether to load the dataset from huggingface or not. Default is False.
        """

        self.relevant_str = relevant[True]
        self.non_relevant_str = relevant[False]
        self.language = language
        self.split = split
        self.load_from_huggingface = load_from_huggingface
        self.hf_dataset_name = hf_dataset_name

        # Initializations of the dataset
        self.corpus = {}
        self.queries = {self.relevant_str: {}, self.non_relevant_str: {}}
        self.qrels = {self.relevant_str: {}, self.non_relevant_str: {}}

        if self.split not in ["dev", "test"]:
            raise ValueError("Split must be one of dev, or test")

        if not load_from_huggingface:
            self._init_path_assignment(corpus_file, data_dir, query_files, qrels_files)

    def _init_path_assignment(self, corpus_file, data_dir, query_files, qrels_files):
        corpus_file = os.path.join(data_dir, corpus_file)

        if query_files is None:
            relevant_query_file = os.path.join(
                data_dir, "topics", f"{self.split}.{self.relevant_str}.tsv"
            )
            non_relevant_query_file = os.path.join(
                data_dir, "topics", f"{self.split}.{self.non_relevant_str}.tsv"
            )
        else:
            relevant_query_file = query_files[self.relevant_str]
            non_relevant_query_file = query_files[self.non_relevant_str]

        if qrels_files is None:
            relevant_qrels_file = os.path.join(
                data_dir, "qrels", f"{self.split}.{self.relevant_str}.tsv"
            )
            non_relevant_qrels_file = os.path.join(
                data_dir, "qrels", f"{self.split}.{self.non_relevant_str}.tsv"
            )
        else:
            relevant_qrels_file = qrels_files[self.relevant_str]
            non_relevant_qrels_file = qrels_files[self.non_relevant_str]

        self.corpus_file = corpus_file
        self.query_files = {
            self.relevant_str: relevant_query_file,
            self.non_relevant_str: non_relevant_query_file,
        }
        self.qrels_files = {
            self.relevant_str: relevant_qrels_file,
            self.non_relevant_str: non_relevant_qrels_file,
        }

    def load_data_from_huggingface(self) -> None:
        """
        Load the NoMIRACL dataset from the huggingface datasets library.
        Each dataset in the NoMIRACL dataset contains the following columns:
        - query_id: The unique identifier of the query.
        - query: The query text.
        - positive_passages: The list of positive passages for the query.
        - negative_passages: The list of negative passages for the query.

        Args:
            dataset_name (str): The name of the dataset to be loaded.
            split (str): The split of the dataset to be used. Default is "test".
        Returns:
            None
        """
        for subset in [self.relevant_str, self.non_relevant_str]:
            dataset_hf = datasets.load_dataset(
                self.hf_dataset_name, self.language, split=f"{self.split}.{subset}"
            )
            for row in dataset_hf:
                query_id = row["query_id"]
                self.queries[subset][query_id] = row["query"]
                if query_id not in self.qrels[subset]:
                    self.qrels[subset][query_id] = {}

                for is_positive, passages in zip(
                    [0, 1], [row["positive_passages"], row["negative_passages"]]
                ):
                    for passage in passages:
                        idx = passage["docid"]
                        if idx not in self.corpus:
                            self.corpus[idx] = {
                                "text": passage["text"],
                                "title": passage["title"],
                            }

                        if idx not in self.qrels[subset][query_id]:
                            self.qrels[subset][query_id][idx] = is_positive
                        else:
                            self.qrels[subset][query_id][idx] = is_positive

    def load_data(self):

        self.check(fIn=self.corpus_file, ext="jsonl")
        self.check(fIn=self.query_files[self.relevant_str], ext="tsv")
        self.check(fIn=self.query_files[self.non_relevant_str], ext="tsv")
        self.check(fIn=self.qrels_files[self.relevant_str], ext="tsv")
        self.check(fIn=self.qrels_files[self.non_relevant_str], ext="tsv")

        if not len(self.corpus):
            logger.info("Loading corpus...")
            self.corpus = load_corpus(filepath=self.corpus_file)
            logger.info("Loaded %d documents.", len(self.corpus))
            logger.info("Doc Example: %s", list(self.corpus.values())[0])

        if not len(self.queries[self.relevant_str]):
            logger.info("Loading relevant queries...")
            queries_relevant = load_queries(
                filepath=self.query_files[self.relevant_str]
            )

        if not len(self.qrels[self.relevant_str]):
            qrels_relevant = load_qrels(filepath=self.qrels_files[self.relevant_str])
            logger.info("Loaded %d relevant queries.", len(qrels_relevant))
            queries_relevant = {
                query_id: query
                for query_id, query in queries_relevant.items()
                if query_id in qrels_relevant
            }

        if not len(self.queries[self.non_relevant_str]):
            logger.info("Loading non-relevant queries...")
            queries_non_relevant = load_queries(
                filepath=self.query_files[self.non_relevant_str]
            )

        if not len(self.qrels[self.non_relevant_str]):
            qrels_non_relevant = load_qrels(
                filepath=self.qrels_files[self.non_relevant_str]
            )
            logger.info("Loaded %d non-relevant queries.", len(qrels_non_relevant))

        # check if query_ids are present in queries and qrels (both)
        qrels_relevant_check, qrels_non_relevant_check = {}, {}
        for query_id in qrels_relevant:
            if query_id in queries_relevant:
                qrels_relevant_check[query_id] = qrels_relevant[query_id]

        for query_id in qrels_non_relevant:
            if query_id in queries_non_relevant:
                qrels_non_relevant_check[query_id] = qrels_non_relevant[query_id]

        self.queries = {
            self.relevant_str: queries_relevant,
            self.non_relevant_str: queries_non_relevant,
        }
        self.qrels = {
            self.relevant_str: qrels_relevant_check,
            self.non_relevant_str: qrels_non_relevant_check,
        }

    def load_data_sample(
        self,
        relevant_ratio: float = 0.5,
        non_relevant_ratio: float = 0.5,
        random_seed: int = 42,
        max_sample_pool: int = None,
    ) -> Tuple[Dict[str, Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:

        # Initialize the sampled queries, and qrels
        sampled_queries = {self.relevant_str: {}, self.non_relevant_str: {}}
        sampled_qrels = {self.relevant_str: {}, self.non_relevant_str: {}}

        # Load the whole dataset either by downloading or using huggingface
        (
            self.load_data()
            if not self.load_from_huggingface
            else self.load_data_from_huggingface()
        )

        # For convinience
        relevant_qrels, relevant_queries = (
            self.qrels[self.relevant_str],
            self.queries[self.relevant_str],
        )
        non_relevant_qrels, non_relevant_queries = (
            self.qrels[self.non_relevant_str],
            self.queries[self.non_relevant_str],
        )

        # Determine whether no-answer or answer is bigger in size and sample accordingly
        relevant_samples = len(relevant_qrels)
        non_relevant_samples = len(non_relevant_qrels)

        # check if either of the relevant or non_relevant ratio are zero
        if relevant_ratio == 0.0 or non_relevant_ratio == 0.0:
            if relevant_ratio == 0.0:
                relevant_samples = 0
            if non_relevant_ratio == 0.0:
                non_relevant_samples = 0

        else:
            # Hypothetical ratios and samples to be sampled
            hyp_non_relevant_samples = int(
                relevant_samples * (non_relevant_ratio / relevant_ratio)
            )
            hyp_relevant_samples = int(
                non_relevant_samples * (relevant_ratio / non_relevant_ratio)
            )

            # Check if the hypothetical relevant samples are greater than the dataset size
            if hyp_relevant_samples > len(relevant_qrels):
                hyp_relevant_samples = len(relevant_qrels)
                non_relevant_samples = int(
                    hyp_relevant_samples * (non_relevant_ratio / relevant_ratio)
                )
                logger.info(
                    "Relevant samples reached max limit: %d", hyp_relevant_samples
                )

            # Check if the samples are greater than the dataset size
            if hyp_non_relevant_samples > len(non_relevant_qrels):
                hyp_non_relevant_samples = len(non_relevant_qrels)
                relevant_samples = int(
                    hyp_non_relevant_samples * (relevant_ratio / non_relevant_ratio)
                )
                logger.info(
                    "Non-relevant samples reached max limit: %d",
                    hyp_non_relevant_samples,
                )

            if (relevant_samples + hyp_non_relevant_samples) > (
                hyp_relevant_samples + non_relevant_samples
            ):
                relevant_samples = relevant_samples
                non_relevant_samples = hyp_non_relevant_samples
            else:
                relevant_samples = hyp_relevant_samples
                non_relevant_samples = non_relevant_samples

        # Check if the samples are greater than the max_sample_pool
        if max_sample_pool is not None:
            if relevant_samples > max_sample_pool:
                relevant_samples = max_sample_pool
            if non_relevant_samples > max_sample_pool:
                non_relevant_samples = max_sample_pool

        # Set the random seed
        random.seed(random_seed)

        # Sample the dataset
        logger.info("Sampling %d relevant queries...", relevant_samples)
        sampled_ids = random.sample(list(relevant_qrels.keys()), relevant_samples)

        for query_id in sampled_ids:
            sampled_qrels["relevant"][query_id] = relevant_qrels[query_id]
            sampled_queries["relevant"][query_id] = relevant_queries[query_id]

        logger.info("Sampling %d non-relevant queries...", non_relevant_samples)
        sampled_ids = random.sample(
            list(non_relevant_qrels.keys()), non_relevant_samples
        )

        for query_id in sampled_ids:
            sampled_qrels["non_relevant"][query_id] = non_relevant_qrels[query_id]
            sampled_queries["non_relevant"][query_id] = non_relevant_queries[query_id]

        return self.corpus, sampled_queries, sampled_qrels
