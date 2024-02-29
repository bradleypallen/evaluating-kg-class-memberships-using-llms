# evaluating-kg-class-memberships-using-llms
 Code and data for experiments on the evaluation of class membership relations in knowledge graphs using LLMs

 [Bradley P. Allen](https://orcid.org/0000-0003-0216-3930) and [Paul T. Groth](https://orcid.org/0000-0003-0183-6910)   
 [INtelligent Data Engineering Lab](https://indelab.org/)  
 University of Amsterdam  
 Amsterdam, The Netherlands

## Overview
A backbone of knowledge graphs are their *class membership relations*, which assign entities to a given class. As part of the knowledge engineering process, we propose a new method for evaluating the quality of these relations by processing descriptions of a given entity and class using a zero-shot chain-of-thought classifier that uses a natural language intensional definition of a class. This repository contains the data and code involved in an evaluation of this method.

We evaluated the method using two publicly available knowledge graphs, Wikidata and CaLiGraph, and 7 large language models. Using the gpt-4-0125-preview large language model, the method’s classification performance achieved a macro-averaged F1-score of 0.830 on data from Wikidata and 0.893 on data from CaLiGraph. Moreover, a manual analysis of the classification errors showed that 40.9% of errors were due to the knowledge graphs, with 16.0% due to missing relations and 24.9% due to incorrectly asserted relations. These results show how large language models can assist knowledge engineers in the process of knowledge graph refinement.

## License
MIT.

## Data sets
- Wikidata: [wikidata_classes.json](wikidata_classes.json)
- CaLiGraph: [caligraph_classes.json](caligraph_classes.json)

## Code
- Classifier implementation: [classifier.py](classifier.py)
- Utilities for running experiments and displaying results: [utils.py](utils.py)

## Experiments
- Notebooks for executing experiments
    - Wikidata: [wikidata_experiment.ipynb](wikidata_experiment.ipynb)
    - CaLiGraph [caligraph_experiment.ipynb](caligraph_experiment.ipynb)
- Experimental results
    - Wikidata
        - gemma-2b-it: [gemma-2b-it-wikidata.json](gemma-2b-it-wikidata.json)
        - gemma-7b-it: [gemma-7b-it-wikidata.json](gemma-7b-it-wikidata.json)
        - gpt-3.5-turbo: [gpt-3.5-turbo-wikidata.json](gpt-3.5-turbo-wikidata.json)
        - gpt-4.0-0125-preview: [gpt-4-0125-preview-wikidata.json](gpt-4-0125-preview-wikidata.json)
        - Llama-2-70b-chat-hf: [Llama-2-70b-chat-hf-wikidata.json](Llama-2-70b-chat-hf-wikidata.json)
        - Mistral-7b-instruct-v0.2: [Mistral-7B-Instruct-v0.2-wikidata.json](Mistral-7B-Instruct-v0.2-wikidata.json)
        - Mixtral-8x7B-Instruct-v0.1: [Mixtral-8x7B-Instruct-v0.1-wikidata.json](Mixtral-8x7B-Instruct-v0.1-wikidata.json)
    - CaLiGraph
        - gemma-2b-it: [gemma-2b-it-caligraph.json](gemma-2b-it-caligraph.json)
        - gemma-7b-it: [gemma-7b-it-caligraph.json](gemma-7b-it-caligraph.json)
        - gpt-3.5-turbo: [gpt-3.5-turbo-caligraph.json](gpt-3.5-turbo-caligraph.json)
        - gpt-4.0-0125-preview: [gpt-4-0125-preview-caligraph.json](gpt-4-0125-preview-caligraph.json)
        - Llama-2-70b-chat-hf: [Llama-2-70b-chat-hf-caligraph.json](Llama-2-70b-chat-hf-caligraph.json)
        - Mistral-7b-instruct-v0.2: [Mistral-7B-Instruct-v0.2-caligraph.json](Mistral-7B-Instruct-v0.2-caligraph.json)
        - Mixtral-8x7B-Instruct-v0.1: [Mixtral-8x7B-Instruct-v0.1-caligraph.json](Mixtral-8x7B-Instruct-v0.1-caligraph.json)

## Results

### Classifier performance
- Wikidata: [wikidata-classifier-performance.ipynb](wikidata-classifier-performance.ipynb)
- CaLiGraph: [caligraph-classifier-performance.ipynb](caligraph-classifier-performance.ipynb)

### Error analysis
- Classifier errors using gpt-4-0125-preview: [gpt-4-0125-preview-errors.ipynb](gpt-4-0125-preview-errors.ipynb)
- Notebook for generating CSV files for import into spreadsheet application in support of human annotation for error analysis: [gpt-4-0125-preview-error-analysis-prep.ipynb](gpt-4-0125-preview-error-analysis-prep.ipynb)
- Generated CSV files for import into spreadsheets for human annotation
    - Wikidata: [wd_err.csv](wd_err.csv)
    - CaLiGraph: [cg_err.csv](cg_err.csv)
- Spreadsheets and CSV files with human annotations:
    - Wikidata: [wd_err_annotated.numbers](wd_err_annotated.numbers) (Numbers), [cg_err_annotated.csv](cg_err_annotated.csv) (CSV)
    - CaLiGraph: []() (Numbers), []() (CSV)
- Error analysis: [gpt-4-0125-preview-error-analysis.ipynb](gpt-4-0125-preview-error-analysis.ipynb)

