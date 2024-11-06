from typing import List
import logging, torch, json, argparse
from sentence_transformers import SentenceTransformer
from model import LLMModel
from llm_predictor import LLMPredictor


def retrieve_topk_vectors_qas(samples: List = None, top_k: int = 3, key_: str = "top_100",
                              output_file:str = "approach_1_dev_top_3.json", progress_top_n: int = 10):
    # https://huggingface.co/Alibaba-NLP/gte-base-en-v1.5 
    model_init = SentenceTransformer('Alibaba-NLP/gte-base-en-v1.5', trust_remote_code=True)
    logging.info("Model to create vector is: Alibaba-NLP/gte-base-en-v1.5")
    
    with open(output_file, "w", encoding="utf-8") as output_json:
        for key in samples:
            sample = samples[key]
            claim, claim_id = sample['claim'], sample['claim_id']
            
            if int(claim_id)%100==0:
                logging.info("Claim: %s", str(claim_id))
            
            ### test
            qas = [i['question'] + " " + i['sentence'] for i in sample[key_][:progress_top_n]]
            sentences = [i['sentence'] for i in sample[key_][:progress_top_n]]
            questions = [i['question'] for i in sample[key_][:progress_top_n]]
            urls = [i['url'] for i in sample[key_][:progress_top_n]]
        
            qas_embeddings = model_init.encode(qas, convert_to_tensor=True)
            query_embedding = model_init.encode(claim, convert_to_tensor=True)
            
            similarity_scores = model_init.similarity(query_embedding, qas_embeddings)[0]
            scores, indices = torch.topk(similarity_scores, k=top_k)
        
            top_k_sentences = [sentences[i] for i in indices]
            top_k_questions = [questions[i] for i in indices]
            top_k_urls = [urls[i] for i in indices]
        
            for i in indices:
                assert sample[key_][:progress_top_n][i]['sentence'] == sentences[i]
                assert sample[key_][:progress_top_n][i]['question'] == questions[i]
                assert sample[key_][:progress_top_n][i]['url'] == urls[i]
                assert sample[key_][:progress_top_n][i]['question'] + " " + sample[key_][:progress_top_n][i]['sentence'] == qas[i]
    
            json_data = {
                "claim_id": claim_id,
                "claim": claim,
                "evidence": [
                    {"answer": s, "question":q, "url": u}
                    for s, q, u in zip(top_k_sentences, top_k_questions, top_k_urls)
                ],
            }
            output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_json.flush()


def compute_veracity_predictions_from_1(sample_file: str = "approach_1_dev_top_5.json",
                                        file_predictions: str = "dev_predictions_approach_1_top5_experiment_1_quantized_chat.json", 
                                        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                        quantized: bool = True, device: int = 3,
                                        prompt_template_init: str = None):
    samples = []
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L87
    with open(sample_file) as f:
        for line in f:
            samples.append(json.loads(line))

    # https://www.promptingguide.ai/models/mixtral, https://www.pinecone.io/learn/mixtral-8x7b/
    # https://huggingface.co/docs/transformers/v4.41.0/en/main_classes/pipelines#transformers.pipeline
    # https://huggingface.co/docs/transformers/main/en/tasks/prompting
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


def compute_veracity_predictions_from_2(sample_file: str = "approach_1_dev_top_5.json",
                                        file_predictions: str = "dev_predictions_approach_1_top5_experiment_2_2_quantized_chat.json", 
                                        model_name: str = "mistralai/Mixtral-8x7B-Instruct-v0.1", 
                                        quantized: bool = True, device: int = 3,
                                        prompt_template_init: str = None, strategy1: bool = False,
                                        start: int = None, end: int = None):
    samples = []
    # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L87
    with open(sample_file) as f:
        for line in f:
            samples.append(json.loads(line))

    # https://www.promptingguide.ai/models/mixtral, https://www.pinecone.io/learn/mixtral-8x7b/
    # https://huggingface.co/docs/transformers/v4.41.0/en/main_classes/pipelines#transformers.pipeline
    # https://huggingface.co/docs/transformers/main/en/tasks/prompting
    # collect predictions
    if not prompt_template_init:
        prompt_template_init = """
            Classify the claim into "Supported" or "Refuted" based on list of evidences. Produce a score for the class label.
            No Explanation, No Note! Your respond should be in JSON format containing `"label"` key-value pair without any further information. For instance, 
            ```json
            {
                "label": "Supported",
                "score": 0.7
            }
            ```
            """
    
    model_init = LLMModel(model_name=model_name, quantized=quantized, device=device)
    predictor = LLMPredictor(prompt_template=prompt_template_init, model=model_init)

    if start != None and end != None:
        samples = samples[start:end]
        logging.info('number of samples to be processed: %s', str(len(samples)))
    predictor.get_predictions_one_evidence_based(samples=samples, file_predictions=file_predictions, 
                                                 qa=True, strategy1=strategy1)


def get_parameters():
    parser = argparse.ArgumentParser(description='Performs re-ranking with QA evidences and LLM experiments')

    llm = parser.add_mutually_exclusive_group(required=True)
    llm.add_argument('--llm', dest='llm', action='store_true',
                     help='determines whether LLM or reranking will be processed')
    llm.add_argument('--re-ranking', dest='llm', action='store_false',
                     help='determines whether LLM or reranking will be processed')
    parser.set_defaults(llm=True)

    parser.add_argument('--input-file', dest='input_file', 
                        default="top10_with_questions.json", # if llm - approach_1_dev_top_10_script.json
                        type=str,
                        help='determines the path for top-k input file')

    parser.add_argument('--topk', dest='top_k', default=10,
                        type=int,
                        help='determines how many evidences to filter')

    parser.add_argument('--key', dest='key', default="top_100",
                        type=str,
                        help='determines key for the input file that is in dictionary format')

    parser.add_argument('--progress-top-n', dest='progress_top_n', default="10",
                        type=int,
                        help='determines how many top n evidences will be re-ranked')

    parser.add_argument('--output-file', dest='output_file', type=str,
                        default='approach_1_dev_top_10.json', # if llm - dev_predictions_approach_1.json
                        help='determines the file path to store the retrieval results')
    
    llm_prompt_1 = parser.add_mutually_exclusive_group(required=False)
    llm_prompt_1.add_argument('--llm-prompt-1', dest='llm_prompt_1', action='store_true',
                              help='determines whether LLM strategy 1 or 2 will be processed')
    llm_prompt_1.add_argument('--llm-prompt-2', dest='llm_prompt_1', action='store_false',
                              help='determines whether LLM strategy 1 or 2 will be processed')
    llm_prompt_1.set_defaults(llm_prompt_1=True)

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
    
    parser.add_argument('--start', dest='start', default=None,
                        type=int,
                        help='start and end are optional to process some pieces of data for llm2')

    parser.add_argument('--end', dest='end', default=None,
                        type=int,
                        help='start and end are optional to process some pieces of data for llm2')

    strategy1 = parser.add_mutually_exclusive_group(required=False)
    strategy1.add_argument('--strategy-1', dest='strategy1', action='store_true',
                           help='determines whether strategy 1 or 2 when LLM 2 was selected will be processed')
    strategy1.add_argument('--strategy-2', dest='strategy1', action='store_false',
                           help='determines whether strategy 1 or 2 when LLM 2 was selected will be processed')
    strategy1.set_defaults(strategy1=False)

    parser.add_argument('--log-file', dest='log_file', default='approach_1_ranking.log', type=str,
                        help='the progress while running the script will be stored in the log file\
                        (default "approach_1_ranking.log")')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parameters()
    logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)
    logging.info(".. reranking and/or LLM experiments are started with args: %s", str(args))

    if not args.llm:
        # re-ranking
        with open(args.input_file, "r", encoding="utf-8") as json_file:
            samples_init = json.load(json_file)
    
        retrieve_topk_vectors_qas(samples=samples_init, top_k=args.top_k, key_= args.key,
                                  progress_top_n = args.progress_top_n, output_file=args.output_file)
        logging.info("%s is finished", str(args.top_k))
        
    else:
        # llm
        if args.llm_prompt_1:
            logging.info("Veracity prediction method-1 for %s, with parameters: model_name - %s, quantized - %s, device - %s", args.input_file, args.model_name, str(args.quantized), str(args.device))
            compute_veracity_predictions_from_1(sample_file = args.input_file,
                                                file_predictions = args.output_file,
                                                model_name = args.model_name, 
                                                quantized = args.quantized, device = args.device)
           
            logging.info("Stored in file %s", args.output_file)
        else:
            logging.info("Veracity prediction method-2 for %s, with parameters: model_name - %s, quantized - %s, device - %s", args.input_file, args.model_name, str(args.quantized), str(args.device))
            compute_veracity_predictions_from_2(sample_file = args.input_file,
                                                file_predictions = args.output_file,
                                                model_name = args.model_name, 
                                                quantized = args.quantized, device = args.device, 
                                                start = args.start, end = args.end, strategy1 = args.strategy1)
            logging.info("Stored in file %s", args.output_file)
    
