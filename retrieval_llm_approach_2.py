from typing import List
import logging, torch, json, argparse
from sentence_transformers import SentenceTransformer
from model import LLMModel
from llm_predictor import LLMPredictor


def retrieve_topk_vectors_qas(samples: List = None, bm25_top10000: List = None, top_k: int = 1, 
                              output_file:str = "approach_2_bm25_10000_dev_top_3_questions_1_answer.json"):
    # https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5 
    model_init = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    logging.info("Model to create vector is: Alibaba-NLP/gte-base-en-v1.5")
    
    with open(output_file, "w", encoding="utf-8") as output_json:
        for key in samples:
            sample = samples[key]
            claim, claim_id = sample['claim'], int(key)
            
            top_10000 = bm25_top10000[claim_id]['top_10000']
    
            assert bm25_top10000[claim_id]['claim'] == claim
            assert bm25_top10000[claim_id]['claim_id'] == claim_id
            
            if claim_id%100 == 0:
                logging.info("Progress: %s/%s", str(claim_id), str(len(samples)))
            
            sentences = [i['sentence'] for i in top_10000]
            urls = [i['url'] for i in top_10000]
            sentence_embeddings = model_init.encode(sentences, convert_to_tensor=True)
            
            evidences = []
            for q in sample['new_questions']:
                # print(claim + " " + q)
                query_embedding = model_init.encode(claim + " " + q, convert_to_tensor=True)
                
                similarity_scores = model_init.similarity(query_embedding, sentence_embeddings)[0]
                scores, indices = torch.topk(similarity_scores, k=top_k)
                # print(similarity_scores, scores, indices)
            
                top_k_sentences = [sentences[i] for i in indices]
                top_k_urls = [urls[i] for i in indices]
            
                # print(top_k_sentences, top_k_urls)
                for i in indices:
                    assert top_10000[i]['sentence'] == sentences[i]
                    assert top_10000[i]['url'] == urls[i]
                
                evidences.extend([{"answer": s, "question":q, "url": u}
                                  for s, u in zip(top_k_sentences, top_k_urls)])
    
            json_data = {
                "claim_id": claim_id,
                "claim": claim,
                "evidence": evidences
            }
            output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_json.flush()


def compute_veracity_predictions_from_1(sample_file: str = "approach_2_bm25_10000_dev_top_3_questions_1_answer_script.json",
                                        file_predictions: str = "dev_predictions_approach_2_top_3_questions_1_experiment_1_quantized_chat.json", 
                                        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                        quantized: bool = True, device: int = 3,
                                        prompt_template_init: str = None):
    samples = []
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L87
    with open(sample_file) as f:
        for line in f:
            samples.append(json.loads(line))

    # collect predictions
    if not prompt_template_init:
        prompt_template_init = """
            Classify the claim into "Supported", "Refuted", "Not Enough Evidence", or "Conflicting Evidence/Cherrypicking" based on list of evidences. 
            No Explanation, No Note! Your respond should be in JSON format containing `"label"` key-value pair without any further information. For instance, 
            ```json
            {
                "label": "Supported"
            }
            ```
            """
    
    model_init = LLMModel(model_name=model_name, quantized=quantized, device=device)
    predictor = LLMPredictor(prompt_template=prompt_template_init, model=model_init)
    
    predictor.get_predictions_with_evidences_QA(samples=samples, file_predictions=file_predictions)

def get_parameters():
    parser = argparse.ArgumentParser(description='Performs LLM experiments and retrieving the \
                                     first sentence per question for Approach 2')

    llm = parser.add_mutually_exclusive_group(required=True)
    llm.add_argument('--llm', dest='llm', action='store_true',
                     help='determines whether LLM or retrieve will be processed')
    llm.add_argument('--retrieve', dest='llm', action='store_false',
                     help='determines whether LLM or retrieve will be processed')
    parser.set_defaults(llm=True)

    parser.add_argument('--input-file', dest='input_file', 
                        default="approach_2_bm25_10000_dev_top_3_questions_1_answer_script.json", # for retrieval, claims_with_questions_3.json
                        type=str,
                        help='determines the path for claims with top 3 questions')

    parser.add_argument('--input-bm25-file', dest='input_bm25_file', 
                        default="bm25_reproduced_dev_top_10000_confirmation.json",
                        type=str,
                        help='determines the path for bm25 ranked top10k samples')

    parser.add_argument('--topk', dest='top_k', default=1,
                        type=int,
                        help='determines how many evidences to retrieve')

    parser.add_argument('--output-file', dest='output_file', type=str,
                        default='dev_predictions_approach_2.json',
                        help='determines the file path to store the retrieval results')

    parser.add_argument('--model-name', dest='model_name', type=str,
                        default='mistralai/Mixtral-8x7B-Instruct-v0.1',
                        help='determines the name of the model for LLM predictions')

    quantized = parser.add_mutually_exclusive_group(required=False)
    quantized.add_argument('--quantized', dest='quantized', action='store_true',
                          help='determines whether quantization will be used for LLM or not')
    quantized.add_argument('--not-quantized', dest='quantized', action='store_false',
                          help='determines whether quantization will be used for LLM or not')
    parser.set_defaults(quantized=True)

    parser.add_argument('--device', dest='device', default=0,
                        type=int,
                        help='determines which GPU device to use')

    parser.add_argument('--log-file', dest='log_file', default='test_approach_1_ranking_1.log', type=str,
                        help='the progress while running the script will be stored in the log file\
                        (default "test_approach_2_ranking_1.log")')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parameters()
    logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)
    logging.info(".. retrieval and/or LLM experiments for Approach 2 are started with args: %s", str(args))
    
    if not args.llm:
        logging.info(".. retrieval for Approach 2 is selected")
        # read questions per claim
        with open(args.input_file, "r", encoding="utf-8") as json_file:
            samples_init = json.load(json_file)

        bm25_top10000_init = []
        with open(args.input_bm25_file, "r", encoding="utf-8") as json_file:
            for line in json_file:
                bm25_top10000_init.append(json.loads(line))

        retrieve_topk_vectors_qas(samples=samples_init, bm25_top10000=bm25_top10000_init, top_k=args.top_k, 
                                  output_file=args.output_file)
    else:
        logging.info(".. LLM for Approach 2 is selected with parameters: model_name %s, quantized %s, device %s", str(args.model_name), str(args.quantized), str(args.device))
    
        compute_veracity_predictions_from_1(sample_file = args.input_file,
                                            file_predictions = args.output_file, 
                                            model_name=args.model_name, 
                                            quantized=args.quantized, device=args.device)
        logging.info("Stored in file %s", args.output_file)

