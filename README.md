# Experiments for Fact-Checking with Real-World Claims
This repository contains an implemention for the paper "UHH at AVeriTeC: RAG for Fact-Checking with Real-World Claims" (link soon), developed for the AVeriTeC shared task (https://fever.ai/task.html).

## Configuration

### 1. Install requirements
(python version used in the experiments is 3.11.9)
```
(optional)
conda create -n rag-averitec-env python=3.11.9
conda activate rag-averitec-env
```
```
pip install -r requirements.txt
```
### 2. Download the data 
- The dataset and knowledge_store are available at https://huggingface.co/chenxwh/AVeriTeC.
- Clone the repository ../ and unzip knowledge_store files or copy the data ../AVeriTeC.

### 3. Make sentence unique (optional)
- Run unique_sentence_creation.py by specifying parameters, e.g.:

```
python unique_sentence_creation.py --input-folder ../AVeriTeC/data_store/knowledge_store/output_dev/ --output-folder knowledge_store_dev_unique_reproduce/ --number-of-files 500 
```

### 4. Select top-10k with BM25
- Run step_1_retrieval_approach_1_2_bm25_vectors.py by specifying parameters, following example for the unique sentence inputs:

```
python step_1_retrieval_approach_1_2_bm25_vectors.py --log-file logs/reproduce_dev_retrieval_bm25_unique.log --retrieve-bm25 --unique-sentence --topk 10000 --start 0 --end 500 --claim-file ../AVeriTeC/data/dev.json --output-file bm25_dev_top_10000_unique_reproduce.json --input-folder knowledge_store_dev_unique_reproduce/
```


- Alternatively, step_1_retrieval_approach_1_2_bm25_vectors.py can be run with original sentence inputs, e.g.:

```
python step_1_retrieval_approach_1_2_bm25_vectors.py --log-file logs/reproduce_dev_retrieval_bm25.log --retrieve-bm25 --not-unique-sentence --topk 10000 --start 0 --end 500 --claim-file ../AVeriTeC/data/dev.json --output-file bm25_dev_top_10000_reproduce.json --input-folder ../AVeriTeC/data_store/knowledge_store/output_dev/
```

- Note that this process takes time.

### 5. Retrieve top-10 with vectorial similarity
-  Run, again, step_1_retrieval_approach_1_2_bm25_vectors.py by specifying parameters, e.g.:

```
python step_1_retrieval_approach_1_2_bm25_vectors.py --log-file logs/reproduce_dev_retrieval_vectors_unique.log --retrieve-vectors --include-scores --topk 10 --output-file bm25_vectors_dev_top_10_unique_reproduce.json --input-file bm25_dev_top_10000_unique_reproduce.json
```

### 6. Gerenerate questions for top-10 sentences
- Use app1_step2_chatgpt_question_generation_on_sentences.py to generate questions.
- input: bm25_vectors_dev_top_10_unique_reproduce.json - output: bm25_vectors_dev_top_10_unique_reproduce_with_questions.json

### 7. Retrieve top-{3,5,7,10} with vectorial similarity of question+answer and claim
- Run reranking_llm_approach_1.py to rerank by specifying parameters, e.g.:

```
python reranking_llm_approach_1.py --log-file logs/reproduce_dev_approach_1_reranking_top10_unique.log  --re-ranking --input-file bm25_vectors_dev_top_10_unique_reproduce_with_questions.json --topk 10 --key top_10 --progress-top-n 10 --output-file approach_1_dev_reranking_top_10_reproduce_unique.json
```

### 8. Get predictions from LLM with the input of claim + top-n question+answer evidences
- Run again reranking_llm_approach_1.py to get predictions from LLM by specifying parameters, e.g.:

```
python reranking_llm_approach_1.py --log-file logs/reproduce_dev_approach_1_llm_1_top10_unique.log --llm --input-file approach_1_dev_reranking_top_10_reproduce_unique.json --llm-prompt-1 --quantized --device 0 --model-name mistralai/Mixtral-8x7B-Instruct-v0.1 --output-file predictions/reproduce_dev_predictions_approach_1_llm1_top10_unique.json
```

### 9. Compute scores
- Scores can be computed using the code provided by task organizers, as explained in - https://huggingface.co/chenxwh/AVeriTeC. For example:
```
cd ../AVeriTeC
```

```
python -m src.prediction.evaluate_veracity --prediction_file ../UHH-at-AVeriTeC/predictions/reproduce_dev_predictions_approach_1_llm1_top10_unique.json --label_file data/dev.json
```

or using the task evaluation page.

### Note for Approaches
- Described pipeline is for Approach-1 (Retrieve-Question), for Approach-2 (Question-Retrieve) see retrieval_llm_approach_2.py file.

### Note for Limitations
- While reproducing the approach from scratch due to the "set" operation in unique sentence creation and the LLM usages, there might be differences.
- For more limitations, please see the paper.

### Note for License
- Disclaimer: before the use, be sure to check licenses of all dependencies, dependencies in the requirements.txt file, and the license of dataset and some code pieces used here, e.g. BM25, provided by task organizers.
