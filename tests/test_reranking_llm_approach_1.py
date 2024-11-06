import random, json, torch
from sentence_transformers import SentenceTransformer


def test_retrieve_topk_vectors_qas():
    # Test: randomly select 100 claims, compute their ranking and check if they are the same as the stored file
    random_indexes = random.sample(list(range(0,500)), k=100)

    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=500

    # new_questions = "top10_with_questions.json"
    new_questions = "bm25_vectors_dev_top_10_unique_reproduce_with_questions.json"
    with open(new_questions, "r", encoding="utf-8") as json_file:
        samples = json.load(json_file)

    model_init = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)

    # top10_file = "approach_1_dev_reranking_top_10_reproduce.json"
    top10_file = "approach_1_dev_reranking_top_10_reproduce_unique.json"
    top_10 = []
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L87
    with open(top10_file) as f:
        for line in f:
            top_10.append(json.loads(line))

    for key in random_indexes: # range(119, 500): # range(0, 500):# 
        sample = samples[str(key)]
        claim, claim_id = sample['claim'], sample['claim_id']
        
        assert top_10[key]['claim'] == claim
        assert top_10[key]['claim_id'] == claim_id
        
        # qas = [i['question'] + " " + i['sentence'] for i in sample['top_100'][:10]]
        # sentences = [i['sentence'] for i in sample['top_100'][:10]]
        # questions = [i['question'] for i in sample['top_100'][:10]]
        # urls = [i['url'] for i in sample['top_100'][:10]]

        qas = [i['question'] + " " + i['sentence'] for i in sample['top_10'][:10]]
        sentences = [i['sentence'] for i in sample['top_10'][:10]]
        questions = [i['question'] for i in sample['top_10'][:10]]
        urls = [i['url'] for i in sample['top_10'][:10]]
    
        qas_embeddings = model_init.encode(qas, convert_to_tensor=True)
        query_embedding = model_init.encode(claim, convert_to_tensor=True)
        
        similarity_scores = model_init.similarity(query_embedding, qas_embeddings)[0]
        scores, indices = torch.topk(similarity_scores, k=10)
    
        for i, e in enumerate(top_10[key]['evidence']):
            assert sentences[indices[i]] == e['answer']
            assert questions[indices[i]] == e['question']
            assert urls[indices[i]] == e['url']

