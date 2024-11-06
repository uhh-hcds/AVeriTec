from step_1_retrieval_approach_1_2_bm25_vectors import combine_all_sentences
import json, random


def test_create_unique_sentence_before_bm25_test1():
    # randomly select 50
    random_indexes = random.sample(list(range(0,500)), k=50)
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=500
    
    for i in random_indexes:
        print(i)
        f_input = "../AVeriTeC/data_store/knowledge_store/output_dev/" + str(i) + ".json"
        f_output = "knowledge_store_dev_unique_reproduce/" + str(i) + ".json"

        sentences, urls, _ = combine_all_sentences(f_input)
        
        unique_knowledge_store = []
        with open(f_output, "r", encoding="utf-8") as json_file:
            for line in json_file:
                unique_knowledge_store.append(json.loads(line))
    
        assert len(unique_knowledge_store) == 1
        sentences_unique = [i['sentence'] for i in unique_knowledge_store[0]['unique']]
        urls_unique = [i['urls'] for i in unique_knowledge_store[0]['unique']]
        
        # test-1: the numbers of set and data should be the same
        assert len(sentences_unique) == len(set(sentences))


def test_create_unique_sentence_before_bm25_test2():
    # randomly select 1 [run time is too much]
    random_indexes = random.sample(list(range(0,500)), k=1)
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=500

    i = random_indexes[0]
    f_input = "../AVeriTeC/data_store/knowledge_store/output_dev/" + str(i) + ".json"
    f_output = "knowledge_store_dev_unique_reproduce/" + str(i) + ".json"

    sentences, urls, _ = combine_all_sentences(f_input)
    
    unique_knowledge_store = []
    with open(f_output, "r", encoding="utf-8") as json_file:
        for line in json_file:
            unique_knowledge_store.append(json.loads(line))

    assert len(unique_knowledge_store) == 1
    sentences_unique = [i['sentence'] for i in unique_knowledge_store[0]['unique']]
    urls_unique = [i['urls'] for i in unique_knowledge_store[0]['unique']]

    # test-2: sentence-url pairs should be in the real data
    sentence_url_real = [(sentence, urls[i]) for i, sentence in enumerate(sentences)]
    for i, sent in enumerate(sentences_unique):
        urls_ = urls_unique[i]
        for url in urls_:
            assert (sent, url) in sentence_url_real

    # test-3: all urls per sentence should be in the processed data
    for sent, url in sentence_url_real:
        index = sentences_unique.index(sent)
        assert url in urls_unique[index]

    # test-4: all sentence-url should be in the processed data
    sentence_url_processed = [(sentence, u) for i, sentence in enumerate(sentences_unique) for u in urls_unique[i]]
    for sent, url in sentence_url_real:
        assert (sent, url) in sentence_url_processed


def test_sort_sentences_urls():
    # randomly select 50 and run the test.
    knowledge_folder = "knowledge_store_dev_unique_reproduce/"

    random_indexes = random.sample(list(range(0,500)), k=50)
    assert len(random_indexes) == len(set(random_indexes))
    for i in random_indexes:
        assert i>=0 
        assert i<=500
    
    for index in random_indexes:
        knowledge_file = knowledge_folder + str(index) + ".json"
        knowledge_file_sorted = knowledge_folder + str(index) + "_sorted.json"
        
        unique_reproduce = []
        with open(knowledge_file, "r", encoding="utf-8") as json_file:
            for line in json_file:
                unique_reproduce.append(json.loads(line))
        
        unique_reproduce_sorted = []
        with open(knowledge_file_sorted, "r", encoding="utf-8") as json_file:
            for line in json_file:
                unique_reproduce_sorted.append(json.loads(line))
        
        assert unique_reproduce[0]['claim_id'] == unique_reproduce_sorted[0]['claim_id']
        
        sentences = [i['sentence'] for i in unique_reproduce[0]['unique']]
        sentences_sorted = [i['sentence'] for i in unique_reproduce_sorted[0]['unique']]
        
        # check sentences
        assert set(sentences) == set(sentences_sorted)
        assert sorted(sentences) == sentences_sorted
        
        sample_number = int(0.01*len(sentences))
        print(sample_number, len(sentences))
        random_indexes = random.sample(list(range(0,len(sentences))), k=sample_number)
        assert len(random_indexes) == len(set(random_indexes))
        
        for i in random_indexes:
            assert i>=0 
            assert i<=len(sentences)
        
        # check urls
        for i in random_indexes:
            s = sentences[i]
            s_sorted_index = sentences_sorted.index(s)
            assert set(unique_reproduce[0]['unique'][i]['urls']) == set(unique_reproduce_sorted[0]['unique'][s_sorted_index]['urls'])
            assert sorted(unique_reproduce[0]['unique'][i]['urls']) == unique_reproduce_sorted[0]['unique'][s_sorted_index]['urls']

