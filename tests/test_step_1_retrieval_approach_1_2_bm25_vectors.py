import random, json, nltk, torch
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


def test_bm25_results():
    # Test-1: no repetitive sentence
    f_bm25_top_10000 = "bm25_dev_top_10000_unique_reproduce.json"
    
    bm25_top_10000 = []
    with open(f_bm25_top_10000, "r", encoding="utf-8") as f:
        for line in f:
            bm25_top_10000.append(json.loads(line))
            
    for sample in bm25_top_10000:
        sentences = [ins['sentence'] for ins in sample['top_10000']]
        assert len(set(sentences)) == len(sentences)
        
    # Test-2: randomly select 10 claims, compute their ranking and check if they are the same as the stored file
    random_indexes = random.sample(list(range(0,500)), k=10)
    
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=500

    claim_file = "../AVeriTeC/data/dev.json"
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L75
    with open(claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)
    
    for index in random_indexes:
        print(index)
        f = "knowledge_store_dev_unique_reproduce/" + str(index) + ".json"
        unique_knowledge_store = []
        with open(f, "r", encoding="utf-8") as json_file:
            for line in json_file:
                unique_knowledge_store.append(json.loads(line))
    
        assert len(unique_knowledge_store) == 1
        sentences = [i['sentence'] for i in unique_knowledge_store[0]['unique']]
        urls = [i['urls'] for i in unique_knowledge_store[0]['unique']]
    
        print(len(sentences), len(urls))
        document = sentences
        query = target_examples[index]['claim']
        tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(query))
        top_k_idx = np.argsort(scores)[::-1]
        print('BM25 finished')
    
        stored_info = bm25_top_10000[index]
        assert stored_info['claim'] == query
    
        for i, info in enumerate(bm25_top_10000[index]['top_10000']):
            assert info['sentence'] == document[top_k_idx[i]]
            assert info['url'] == urls[top_k_idx[i]]
    

def test_bm25_results_with_scores():
    # Test: randomly select 50 claims, compute ranking and check if they are the same as the stored file -- with scores
    random_indexes = random.sample(list(range(0,2215)), k=50)
    
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=2215

    f_bm25_top_10000 = "bm25_test_top_10000_unique_reproduce.json"
    bm25_top_10000 = []
    with open(f_bm25_top_10000, "r", encoding="utf-8") as f:
        for line in f:
            bm25_top_10000.append(json.loads(line))
    assert len(bm25_top_10000) == 2215

    claim_file = "../AVeriTeC/data/test.json"
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L75
    with open(claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)
    
    unique_sentences_folder = "knowledge_store_test_unique_reproduce/"
    top_k = 10000
    for index in random_indexes:
        unique_sentences_file = unique_sentences_folder + str(index) + ".json"
    
        unique_knowledge_store = []
        with open(unique_sentences_file, "r", encoding="utf-8") as json_file:
            for line in json_file:
                unique_knowledge_store.append(json.loads(line))
        
        assert len(unique_knowledge_store) == 1
        sentences = [i['sentence'] for i in unique_knowledge_store[0]['unique']]
        urls = [i['urls'] for i in unique_knowledge_store[0]['unique']]
    
        document = sentences
        query = target_examples[index]['claim']
        
        tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
        bm25 = BM25Okapi(tokenized_docs)
        scores = bm25.get_scores(nltk.word_tokenize(query))
        top_k_idx = np.argsort(scores)[::-1][:top_k]
     
        stored_info = bm25_top_10000[index]
        
        assert stored_info['claim'] == query
        assert stored_info['claim_id'] == target_examples[index]['claim_id']
    
        for i, info in enumerate(stored_info['top_10000']):
            if i < 3:
                print(info['sentence'])
            assert info['sentence'] == document[top_k_idx[i]]
            assert info['url'] == urls[top_k_idx[i]]
            assert info['score'] == scores[top_k_idx[i]]
    

def test_combination_results_with_scores():
    # Test: randomly select 50 claims, compute ranking both vector and bm25 and check if they are the same as the stored file -- with scores
    random_indexes = random.sample(list(range(0,2215)), k=50)
    
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=2215

    f_bm25_top_10000 = "bm25_test_top_10000_unique_reproduce.json"
    bm25_top_10000 = []
    with open(f_bm25_top_10000, "r", encoding="utf-8") as f:
        for line in f:
            bm25_top_10000.append(json.loads(line))
    assert len(bm25_top_10000) == 2215

    f_combination = "bm25_vectors_test_top_10_unique_reproduce.json"
    combination = []
    with open(f_combination, "r", encoding="utf-8") as f:
        for line in f:
            combination.append(json.loads(line))
    
    # https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5 
    model_init = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

    for index in random_indexes:
        sample = bm25_top_10000[index]
        stored_info = combination[index]
        
        s = [i["sentence"] for i in sample["top_10000"]]
        u = [i["url"] for i in sample["top_10000"]]
        bm25 = [i["score"] for i in sample["top_10000"]]
    
        claim = sample["claim"]
        assert claim == stored_info["claim"]
        assert sample["claim_id"] == stored_info["claim_id"]
    
        sentences_embeddings = model_init.encode(s, convert_to_tensor=True)
        query_embedding = model_init.encode(claim, convert_to_tensor=True)
        
        similarity_scores = model_init.similarity(query_embedding, sentences_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=10)
        
        for i_, i in enumerate(indices):           
            assert bm25[i] == stored_info["top_10"][i_]["score_bm25"]
            assert round(float(scores[i_]), 2) == round(stored_info["top_10"][i_]["score_vectors"], 2)
            assert s[i] == stored_info["top_10"][i_]["sentence"]
            assert u[i] == stored_info["top_10"][i_]["url"]

