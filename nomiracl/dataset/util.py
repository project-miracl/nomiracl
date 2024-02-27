import csv
import json
from tqdm.autonotebook import tqdm


def load_queries(filepath: str):
    """
    Loads queries from a file and stores them in a dictionary.
    """
    queries = {}
    num_lines = sum(1 for i in open(filepath, 'rb'))
    with open(filepath, 'r', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
        for row in tqdm(reader, total=num_lines, desc="Loading Queries"):
            queries[row[0]] = row[1]
    return queries

def load_qrels(filepath: str):
    """
    Loads the qrels file as a dictionary.
    """
    qrels = {}
    reader = csv.reader(open(filepath, encoding="utf-8"), 
                        delimiter="\t", quoting=csv.QUOTE_MINIMAL)
    
    for _, row in enumerate(reader):
        query_id, corpus_id, score = row[0], row[2], int(row[3])
        
        if query_id not in qrels:
            qrels[query_id] = {corpus_id: score}
        else:
            qrels[query_id][corpus_id] = score
    return qrels

def load_corpus(filepath: str):
    """
    Loads the corpus file as a dictionary.
    """
    corpus = {}
    num_lines = sum(1 for i in open(filepath, 'rb'))
    with open(filepath, encoding='utf8') as fIn:
        for line in tqdm(fIn, total=num_lines, desc="Loading Corpus"):
            line = json.loads(line)
            corpus[line.get("docid")] = {
                "text": line.get("text").strip().replace("\n", " "),
                "title": line.get("title").strip().replace("\n", " "),
            }
    return corpus