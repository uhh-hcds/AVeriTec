# Disclaimer -- original code is here: https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py
import json, os, time, nltk, logging, torch, argparse
import numpy as np
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L10
def combine_all_sentences(knowledge_file):
    sentences, urls = [], []

    with open(knowledge_file, "r", encoding="utf-8") as json_file:
        for i, line in enumerate(json_file):
            data = json.loads(line)
            sentences.extend(data["url2text"])
            urls.extend([data["url"] for i in range(len(data["url2text"]))])
    return sentences, urls, i + 1


# https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L21
def retrieve_top_k_sentences(query, document, urls, top_k):
    tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(nltk.word_tokenize(query))
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx]


# https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L21
def retrieve_top_k_sentences_with_scores(query, document, urls, top_k):
    tokenized_docs = [nltk.word_tokenize(doc) for doc in document]
    bm25 = BM25Okapi(tokenized_docs)
    scores = bm25.get_scores(nltk.word_tokenize(query))
    top_k_idx = np.argsort(scores)[::-1][:top_k]

    return [document[i] for i in top_k_idx], [urls[i] for i in top_k_idx], [scores[i] for i in top_k_idx]


def retrieve_bm25_topk(output_file:str = "bm25_reproduced_dev_top_10000.json",
                       knowledge_store_dir = "../AVeriTeC/data_store/knowledge_store/output_dev/",
                       claim_file = "../AVeriTeC/data/dev.json", 
                       top_k:int = 10000, start:int = 0, end:int = 500):
    
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L75
    with open(claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)

    logging.info("Started bm25 scores with start: %s and end: %s", str(start), str(end))
    
    files_to_process = list(range(start, end))
    total = len(files_to_process)
    
    with open(output_file, "a", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            # Load the knowledge store for this example
            if idx in files_to_process:
                logging.info("Processing claim %s.. Progress: %s/%s", str(idx), str(done + 1), str(total))
                document_in_sentences, sentence_urls, num_urls_this_claim = (
                    combine_all_sentences(
                        os.path.join(knowledge_store_dir, f"{idx}.json")
                    )
                )
    
                logging.info("Obtained %s sentences from %s urls.", str(len(document_in_sentences)), str(num_urls_this_claim))
    
                # Retrieve top_k sentences with bm25
                st = time.time()
                top_k_sentences, top_k_urls = retrieve_top_k_sentences(
                    example["claim"], document_in_sentences, sentence_urls, top_k
                )
                logging.info("Top %s retrieved. Time elapsed: %s.", str(top_k), str(time.time() - st))
    
                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{top_k}": [
                        {"sentence": sent, "url": url}
                        for sent, url in zip(top_k_sentences, top_k_urls)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1
                output_json.flush()


def read_sentences_urls(f: str):
    unique_knowledge_store = []
    with open(f, "r", encoding="utf-8") as json_file:
        for line in json_file:
            unique_knowledge_store.append(json.loads(line))

    # print(len(unique_knowledge_store))
    assert len(unique_knowledge_store) == 1
    sentences = [i['sentence'] for i in unique_knowledge_store[0]['unique']]
    urls = [i['urls'] for i in unique_knowledge_store[0]['unique']]
    return sentences, urls, unique_knowledge_store[0]['claim_id']

    
def retrieve_bm25_topk_from_unique(output_file:str = "bm25_test_top_10000_unique.json",
                                   unique_sentences_folder:str = "knowledge_store_dev_unique/",
                                   claim_file = "../AVeriTeC/data/dev.json",
                                   top_k:int = 10000, start:int = 0, end:int = 500):
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/reranking/bm25_sentences.py#L75
    with open(claim_file, "r", encoding="utf-8") as json_file:
        target_examples = json.load(json_file)
    
    files_to_process = list(range(start, end))
    total = len(files_to_process)
    logging.info('start index %s, end index %s', str(start), str(end))
        
    # with open(output_file, "a", encoding="utf-8") as output_json:
    with open(output_file, "a", encoding="utf-8") as output_json:
        done = 0
        for idx, example in enumerate(target_examples):
            # Load the knowledge store for this example
            if idx in files_to_process:
                if idx%100 == 0:
                    logging.info("Processing claim %s.. Progress: %s/%s", str(idx), str(done + 1), str(total))

                document_in_sentences, sentence_urls, claim_id_file = read_sentences_urls(f=unique_sentences_folder+str(idx)+".json")
                assert int(claim_id_file) == idx
                # logging.info("Obtained %s sentences from %s urls.", str(len(document_in_sentences)), str(num_urls_this_claim))
    
                # Retrieve top_k sentences with bm25
                st = time.time()
                top_k_sentences, top_k_urls, top_k_scores = retrieve_top_k_sentences_with_scores(
                    example["claim"], document_in_sentences, sentence_urls, top_k
                )
                logging.info("Top %s retrieved. Time elapsed: %s.", str(top_k), str(time.time() - st))
    
                json_data = {
                    "claim_id": idx,
                    "claim": example["claim"],
                    f"top_{top_k}": [
                        {"sentence": sent, "url": url, "score": score}
                        for sent, url, score in zip(top_k_sentences, top_k_urls, top_k_scores)
                    ],
                }
                output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
                done += 1
                output_json.flush()
                

def retrieve_vectors_topk(file_bm25_top_10000:str = "bm25_reproduced_dev_top_10000.json", 
                          top_k:int = 10, file_output: str = "combination_bm25_10000_vectors_dev_top_10.json",
                          include_scores=False):
    bm25_top_10000 = []
    with open(file_bm25_top_10000, "r", encoding="utf-8") as f:
        for line in f:
            bm25_top_10000.append(json.loads(line))

    logging.info("Number of bm25 samples: %s", str(len(bm25_top_10000)))

    # https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5 
    model_init = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    logging.info("Model to create vector is: Alibaba-NLP/gte-base-en-v1.5")

    with open(file_output, "w", encoding="utf-8") as output_json:
        for index, sample in enumerate(bm25_top_10000):
            claim = sample["claim"]
            
            # https://sbert.net/examples/applications/semantic-search/README.html
            s = [i["sentence"] for i in sample["top_10000"]]
            u = [i["url"] for i in sample["top_10000"]]
            if include_scores and "score" in sample["top_10000"][0]:
                sc = [i["score"] for i in sample["top_10000"]]
            
            if index%50==0:
                logging.info("Processing claim %s.. Progress: %s/%s", str(index), str(index + 1), str(len(bm25_top_10000)))
                
            sentences_embeddings = model_init.encode(s, convert_to_tensor=True)
            query_embedding = model_init.encode(claim, convert_to_tensor=True)
            
            similarity_scores = model_init.similarity(query_embedding, sentences_embeddings)[0]
            scores, indices = torch.topk(similarity_scores, k=top_k)
        
            sentences = [s[i] for i in indices]
            urls = [u[i] for i in indices]
            scores_bm25 = None
            if include_scores and "score" in sample["top_10000"][0]:
                scores_bm25 = [sc[i] for i in indices]
        
            for i,j in enumerate(indices):
                assert sample['top_10000'][j]['sentence'] == sentences[i]
                assert sample['top_10000'][j]['url'] == urls[i]
                if include_scores and "score" in sample["top_10000"][0]:
                    assert sample['top_10000'][j]['score'] == scores_bm25[i]

            if include_scores and scores_bm25:
                json_data = {
                    "claim_id": sample["claim_id"],
                    "claim": claim,
                    f"top_{top_k}": [
                        {"sentence": s, "url": u, "score_bm25": sc_bm25, "score_vectors": float(sc_vector)}
                        for s, u, sc_bm25, sc_vector in zip(sentences, urls, scores_bm25, scores)
                    ],
                }
            elif include_scores:
                json_data = {
                    "claim_id": sample["claim_id"],
                    "claim": claim,
                    f"top_{top_k}": [
                        {"sentence": s, "url": u, "score_vectors": float(sc_vector)}
                        for s, u, sc_vector in zip(sentences, urls, scores)
                    ],
                }
            else:
                json_data = {
                        "claim_id": sample["claim_id"],
                        "claim": claim,
                        f"top_{top_k}": [
                            {"sentence": s, "url": u}
                            for s, u in zip(sentences, urls)
                        ],
                    }
            output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_json.flush()


def get_parameters():
    parser = argparse.ArgumentParser(description='Performs retrieval with BM25 and vector similarity.')
    
    retrieval_bm25 = parser.add_mutually_exclusive_group(required=False)
    retrieval_bm25.add_argument('--retrieve-bm25', dest='retrieval_bm25', action='store_true',
                                 help='determines whether retrieve top-10k with bm25 or retrieve\
                                 from vector similarity based on given bm25')
    retrieval_bm25.add_argument('--retrieve-vectors', dest='retrieval_bm25', action='store_false',
                                 help='determines whether retrieve top-10k with bm25 or retrieve\
                                 from vector similarity based on given bm25')
    parser.set_defaults(retrieval_bm25=True)

    parser.add_argument('--input-folder', dest='input_folder', 
                        default="knowledge_store_dev_unique/", # for test "knowledge_store_test_unique/"
                        # if knowledge_dir: test - ../AVeriTeC/data_store/knowledge_store/test/ or 
                        # dev - ../AVeriTeC/data_store/knowledge_store/output_dev/
                        type=str,
                        help='determines the path for the unique sentences folder or the knowledge dir')
    
    parser.add_argument('--output-file', dest='output_file', 
                        default='bm25_dev_top_10000_unique.json', type=str,
                        help='determines the file path to store the retrieval results')

    parser.add_argument('--input-file', dest='input_file', 
                        default="bm25_dev_top_10000_unique.json",
                        type=str,
                        help='determines the path for bm25 top-k to use it vector ranking')

    parser.add_argument('--claim-file', dest='claim_file', 
                        default='../AVeriTeC/data/dev.json', # for test ../AVeriTeC/data/test.json
                        type=str,
                        help='determines the file path to store the retrieval results')

    parser.add_argument('--topk', dest='top_k', default=10000, # for vector 100
                        type=int,
                        help='determines how many sentences to retrieve')

    parser.add_argument('--start', dest='start', 
                        default=0,
                        type=int,
                        help='determines which file to start processing')

    parser.add_argument('--end', dest='end', 
                        default=500, # for test 2215
                        type=int,
                        help='determines which file to stop processing')

    unique_sentence = parser.add_mutually_exclusive_group(required=False)
    unique_sentence.add_argument('--unique-sentence', dest='unique_sentence', action='store_true',
                                 help='determines the given folder contains unique sentence or not')
    unique_sentence.add_argument('--not-unique-sentence', dest='unique_sentence', action='store_false',
                                 help='determines the given folder contains unique sentence or not')
    parser.set_defaults(unique_sentence=True)

    
    include_scores = parser.add_mutually_exclusive_group(required=False)
    include_scores.add_argument('--include-scores', dest='include_scores', action='store_true',
                                 help='determines whether scores of bm25 and vectors will be stored or not')
    include_scores.add_argument('--not-include-scores', dest='include_scores', action='store_false',
                                 help='determines whether scores of bm25 and vectors will be stored or not')
    parser.set_defaults(include_scores=False)

    parser.add_argument('--log-file', dest='log_file', default='test_retrieval_bm25_vectors.log', type=str,
                        help='the progress while running the script will be stored in the log file\
                        (default "test_retrieval_bm25_vectors.log")')
    
    return parser.parse_args()


if __name__ == '__main__':    
    args = get_parameters()
    logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)

    logging.info(".. retrieval is started with args: %s", str(args))
    
    if args.retrieval_bm25:
        if args.unique_sentence:
            logging.info(".. started retrieve_bm25_topk_from_unique")
            retrieve_bm25_topk_from_unique(output_file = args.output_file,
                                           unique_sentences_folder = args.input_folder,
                                           claim_file = args.claim_file,
                                           top_k = args.top_k, start = args.start, end = args.end)
        
        else:
            logging.info(".. started retrieve_bm25_topk")
            retrieve_bm25_topk(output_file = args.output_file,
                               claim_file = args.claim_file,
                               knowledge_store_dir = args.input_folder,
                               top_k=args.top_k, start = args.start, end = args.end)
    else: 
        logging.info(".. started retrieve_vectors_topk")
        retrieve_vectors_topk(file_bm25_top_10000 = args.input_file,
                              top_k=args.top_k,
                              file_output = args.output_file, 
                              include_scores=args.include_scores)

