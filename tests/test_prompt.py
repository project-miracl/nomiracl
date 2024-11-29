from __future__ import annotations

import pytest

from nomiracl.prompts.utils import load_prompt_template

@pytest.fixture
def test_data():
    """Fixture for test data."""
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
    expected_output = (
        "You are an evaluator checking whether the question contains the answer within the contexts or not. "
        "I will give you a question and several contexts containing information about the question. "
        "Read the contexts carefully. If any of the contexts answers the question, respond as either "
        "\"Yes, answer is present\" or \"I don't know\". Do not add any other information in your output.\n\n"
        "QUESTION:\nWhich is the best programming language?\n\n"
        "CONTEXTS:\n"
        "[1] Python is the best programming language.\n"
        "[2] Javascript is the best programming language.\n"
        "[3] Go is the best programming language.\n"
        "[4] Java is the best programming language.\n"
        "[5] C# is the best programming language.\n"
        "[6] Ruby is the best programming language.\n"
        "[7] R is the best programming language.\n"
        "[8] C++ is the best programming language.\n"
        "[9] C is the best programming language.\n"
        "[10] Rust is the best programming language.\n\n"
        "OUTPUT:"
    )
    return query, passages, expected_output

def test_prompt_generation(test_data):
    """Test if the prompt template generates the correct output."""
    query, passages, expected_output = test_data

    # Load the prompt template
    prompt_cls = load_prompt_template("role", count=10)
    
    # Generate the prompt
    prompt = prompt_cls(query=query, passages=passages)
    
    # Assert that the generated prompt matches the expected output
    assert prompt.strip() == expected_output.strip()
