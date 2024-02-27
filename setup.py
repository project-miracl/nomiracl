from setuptools import setup, find_packages

with open("README.md", mode="r", encoding="utf-8") as readme_file:
    readme = readme_file.read()

setup(
    name='nomiracl',
    version='0.0.1',
    author="Nandan Thakur",
    author_email="nandant@gmail.com",
    description='Unanswerable questions for LLM hallucations',
    long_description=readme,
    long_description_content_type="text/markdown",
    license="Apache License 2.0",
    url='',
    download_url="",
    packages=find_packages(),
    python_requires='>=3.8',
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    install_requires=[
        "torch",
        "transformers",
        "openai",
        "tiktoken",
        "bitsandbytes",
        "datasets"
    ],
    keywords="Transformer Networks BERT PyTorch NLP deep learning LLM Hallucination"
)