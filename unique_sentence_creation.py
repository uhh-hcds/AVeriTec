from typing import List
from collections import defaultdict
import logging, json, argparse
from step_1_retrieval_approach_1_2_bm25_vectors import combine_all_sentences


def create_unique_sentence_before_bm25(folder: str = "../AVeriTeC/data_store/knowledge_store/output_dev/", 
                                       end: int = 500, 
                                       output_folder: str = "knowledge_store_dev_unique/"): 
    for index in range(end):
        if index%100==0:
            logging.info("Process: %s/%s", str(index), str(end))
        f = folder + str(index) + ".json"
        sentences, urls, _ = combine_all_sentences(f)
        
        output_file = output_folder + str(index) + ".json"
        with open(output_file, "w", encoding="utf-8") as output_json:            
            # sentence-url pairs
            s_u = set()
            for index, s in enumerate(sentences):
                s_u.add((s, urls[index]))
            
            d = defaultdict(list)
            for sent, url in s_u:
                d[sent].append(url)

            claim_id = None
            with open(f, "r", encoding="utf-8") as json_file:
                for i, line in enumerate(json_file):
                    data = json.loads(line)
                    if not claim_id:
                        claim_id = data['claim_id']
                    else:
                        assert data['claim_id'] == claim_id
                
            json_data = {
                    "claim_id": claim_id,
                    "unique": [
                        {"sentence": sentence, "urls": urls}
                        for sentence, urls in d.items()
                    ],
                }
            output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_json.flush()


def sort_sentences_urls(knowledge_folder: str = "knowledge_store_dev_unique/", 
                        end: int = 500):
    for index in range(end):
        if index%100==0:
            logging.info("Process: %s/%s", str(index), str(end))
        
        output_file = knowledge_folder + str(index) + "_sorted.json"
        knowledge_file = knowledge_folder + str(index) + ".json"
        with open(output_file, "w", encoding="utf-8") as output_json:            
            unique = []
            with open(knowledge_file, "r", encoding="utf-8") as json_file:
                for line in json_file:
                    unique.append(json.loads(line))
            assert len(unique) == 1
    
            # sort by sentence
            sentence_urls = [[i["sentence"], i["urls"]]  for i in unique[0]["unique"]]
            sentence_urls_sorted = sorted(sentence_urls, key=lambda x: x[0])
    
            # sort urls per sentence if multiple
            for j, s_u in enumerate(sentence_urls_sorted):
                # sort urls
                if len(s_u[1]) > 1:
                    s_u[1] = sorted(s_u[1])
                
            json_data = {
                    "claim_id": unique[0]["claim_id"],
                    "unique": [
                        {"sentence": sentence, "urls": urls}
                        for sentence, urls in sentence_urls_sorted
                    ],
                }
            output_json.write(json.dumps(json_data, ensure_ascii=False) + "\n")
            output_json.flush()


def get_parameters():
    parser = argparse.ArgumentParser(description='Performs the creation of unique sentences from \
                                     sentences in knowledge store provided by AVeriTeC Shared Task (Schlichtkrull et al., 2023).')
    
    parser.add_argument('--input-folder', dest='input_folder', 
                        default="../AVeriTeC/data_store/knowledge_store/output_dev/",
                        # for test "../AVeriTeC/data_store/knowledge_store/test/"
                        type=str,
                        help='determines the path for the knowledge store folder') 
    
    parser.add_argument('--output-folder', dest='output_folder', 
                        default="knowledge_store_dev_unique/", # for test "knowledge_store_test_unique/"
                        type=str,
                        help='determines the path for the output folder to store unique sentences')
    
    parser.add_argument('--number-of-files', dest='end', 
                        default=500, # for test 2215
                        type=int,
                        help='determines how many files exist in the folder')

    apply_sort = parser.add_mutually_exclusive_group(required=False)
    apply_sort.add_argument('--apply-sort', dest='apply_sort', action='store_true',
                            help='determines if the unique sentences will be sorted or not (default False)')
    apply_sort.add_argument('--not-apply-sort', dest='apply_sort', action='store_false',
                            help='determines if the unique sentences will be sorted or not (default False)')
    parser.set_defaults(apply_sort=False)

    parser.add_argument('--log-file', dest='log_file', default='unique_sentence_creation_before_bm25.log', type=str,
                        help='the progress while running the script will be stored in the log file\
                        (default "unique_sentence_creation_before_bm25.log")')
    
    return parser.parse_args()


if __name__ == "__main__":
    args = get_parameters()
    logging.basicConfig(filename=args.log_file, format='%(asctime)s - %(name)s - %(message)s', level=logging.DEBUG)
    
    logging.info(".. unique sentence creation before bm25 is started with args: %s", args)
    if args.apply_sort:
        sort_sentences_urls(knowledge_folder = args.output_folder, 
                            end = args.end)
    else:
        create_unique_sentence_before_bm25(folder = args.input_folder, 
                                           end = args.end, 
                                           output_folder = args.output_folder)


