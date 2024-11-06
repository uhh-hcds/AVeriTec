from model import LLMModel
import json, logging, torch
from typing import List
from collections import Counter
from transformers import set_seed


class LLMPredictor:
    def __init__(self, prompt_template:str, model: LLMModel, chat_samples:List = None):
        self.prompt_template = prompt_template
        self.llm_model = model
        self.chat_samples = chat_samples

    # one or more evidences
    def get_prediction_per_claim_evidence_s(self, claim, evidence, claim_id=None, log_prompt=False, length_chat=6):
        label_ = "\nClass: "
        if not self.chat_samples:
            # https://huggingface.co/docs/transformers/main/en/chat_templating
            chat = [
                {"role": "user", "content": f"{self.prompt_template}{claim}{evidence}{label_}"},
            ]
        else:
            assert len(self.chat_samples) == length_chat
            chat = [s for s in self.chat_samples]
            chat.append({"role": "user", "content": f"{self.prompt_template}{claim}{evidence}{label_}"})
            
        chat_template = self.llm_model.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        if log_prompt:
            logging.info('Sample of prompt: %s', str(chat_template))
        generated = self.llm_model.generate_text(chat_template)

        try:
            response = json.loads(generated.strip())
            label = response['label']
            return response
        except:
            logging.error('Claim id: %s -- JSON Error: %s', str(claim_id), 
                          generated.strip())
            return None

    def load_claims(self, file_claims_: str = "../AVeriTeC/data_store/dev_top_3_rerank_qa.json"):
        samples_ = []
        # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L87
        with open(file_claims_) as f:
            for line in f:
                samples_.append(json.loads(line))
        return samples_

    def get_predictions_with_evidences_QA(self, samples: List = None, file_predictions: str = None, 
                                          length_chat: int =6):
        count_default = 0
        predictions = []
        for i, sample in enumerate(samples):
            evidences = [evidence['question'] + ' ' + evidence['answer'] for evidence in sample["evidence"]]
            claim_id = sample['claim_id']
            
            claim_ = '\nUser Claim: ' + sample['claim']
            evidences_ = '\nEvidences: ' + str(evidences)
            
            if i==0:
                prediction = self.get_prediction_per_claim_evidence_s(claim=claim_, evidence=evidences_, 
                                                                      claim_id=claim_id, log_prompt=True, 
                                                                      length_chat=length_chat)
            else:
                prediction = self.get_prediction_per_claim_evidence_s(claim=claim_, evidence=evidences_, 
                                                                      claim_id=claim_id, log_prompt=False, 
                                                                      length_chat=length_chat)
            
            if not prediction or prediction['label'] not in ["Supported", "Refuted", "Not Enough Evidence", 
                                                             "Conflicting Evidence/Cherrypicking"]:
                prediction_label = "Refuted" # most common in training
                logging.info('Claim id: %s -- assigned "Refuted"', str(claim_id))
                count_default += 1
            else:
                prediction_label = prediction['label']
                
            prediction_evidences = sample["evidence"]
            # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L158
            json_data = {
                "claim_id": sample["claim_id"],
                "claim": sample["claim"],
                "evidence": prediction_evidences,
                "pred_label": prediction_label,
            }
            predictions.append(json_data)
        if file_predictions:
            # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L166
            with open(file_predictions, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
        logging.info('Number of predictions that default is assigned: %s', str(count_default))
        return predictions

    def get_predictions_one_evidence_based(self, samples: List = None, file_predictions: str = None, 
                                           qa: bool = False, strategy1: bool = True):
        ## strategy1: assign Conflicting Evidence/Cherrypicking if not all agree
        ## strategy2: assign majorities decision and Conflicting Evidence/Cherrypicking if all are different labels
        s = "Strategy 1" if strategy1 else "Strategy 2"
        logging.info('Strategy is selected: %s', s)
        
        count_default, count_agree, count_assigned, count_majority = 0, 0, 0, 0
        predictions = []
        for i, sample in enumerate(samples):
            claim_id = sample['claim_id']
            prediction_labels = []
            
            for evidence in sample["evidence"]:
                claim_ = '\nUser Claim: ' + sample['claim']
                if qa:
                    evidence_ = '\nEvidence: ' + str(evidence['question'] + ' ' + evidence['answer'])
                else:
                    evidence_ = '\nEvidence: ' + str(evidence['answer'])
                
                if i==0:
                    response = self.get_prediction_per_claim_evidence_s(claim=claim_, evidence=evidence_, 
                                                                        claim_id=claim_id, log_prompt=True)
                else:
                    response = self.get_prediction_per_claim_evidence_s(claim=claim_, evidence=evidence_, 
                                                                        claim_id=claim_id, log_prompt=False)

                if not response:
                    prediction_labels.append("Refuted") # most common in training
                    logging.info('Claim id: %s -- assigned "Refuted" with score: 1.0', str(claim_id))
                    count_default += 1
                    
                elif response['label'] in ["Refuted", "Supported"]:
                    prediction_labels.append(response['label'])
                    
                elif 'score' in response and response['score'] and response['score'] <= 0.5:
                    prediction_labels.append("Not Enough Evidence")
                    logging.info('Claim id: %s -- assigned "Not Enough Evidence" with score: %s', str(claim_id), str(response['score']))
                    
                else:
                    prediction_labels.append("Refuted")
                    s = 0.0
                    if 'score' in response:
                        s = response['score']
                    logging.info('Claim id: %s -- assigned "Refuted" for response label (%s) and score: %s', str(claim_id), str(response['label']), str(s))
                    count_default += 1

            prediction = None
            if strategy1:
                if len(set(prediction_labels)) == 1: # if all agree
                    prediction = prediction_labels[0]
                    count_agree += 1
                    assert prediction_labels[0] == prediction_labels[1]
                    assert prediction_labels[0] == prediction_labels[2]
                    
                else:
                    # assign "Conflicting Evidence/Cherrypicking"
                    prediction = "Conflicting Evidence/Cherrypicking"
                    logging.info('Claim id: %s -- assigned "Conflicting Evidence/Cherrypicking" with prediction labels: %s', str(claim_id), str(prediction_labels))
                    count_assigned += 1
            else: # strategy 2 
                if len(set(prediction_labels)) == 1: # if all agree
                    prediction = prediction_labels[0]
                    count_agree += 1
                    assert prediction_labels[0] == prediction_labels[1]
                    assert prediction_labels[0] == prediction_labels[2]
                else:
                    frequencies = Counter(prediction_labels)
                    if len(frequencies.keys()) < 3: # Supported, Refuted, or Not Enough Evidence
                        prediction = max(frequencies, key=frequencies.get)
                        logging.info('Claim id: %s -- majority label assigned: %s, with prediction labels: %s', str(claim_id), prediction, str(prediction_labels))
                        count_majority += 1
                    else: # all different labels
                        # assign "Conflicting Evidence/Cherrypicking"
                        prediction = "Conflicting Evidence/Cherrypicking"
                        logging.info('Claim id: %s -- assigned "Conflicting Evidence/Cherrypicking" with prediction frquencies: %s and prediction labels: %s', str(claim_id), str(frequencies), str(prediction_labels))
                        count_assigned += 1
                            
            assert prediction
            
            prediction_evidences = sample["evidence"]
            # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L158
            json_data = {
                "claim_id": sample["claim_id"],
                "claim": sample["claim"],
                "evidence": prediction_evidences,
                "pred_label": prediction,
            }
            predictions.append(json_data)
        if file_predictions:
            # https://huggingface.co/chenxwh/AVeriTeC/blob/main/src/prediction/veracity_prediction.py#L166
            with open(file_predictions, "w", encoding="utf-8") as f:
                json.dump(predictions, f, ensure_ascii=False, indent=4)
        logging.info('Number of predictions that default is assigned: %s', str(count_default))
        logging.info('Number of predictions that all agree: %s', str(count_agree))
        logging.info('Number of predictions that "Conflicting Evidence/Cherrypicking" is assigned: %s', str(count_assigned))
        logging.info('Number of predictions that majority is assigned: %s', str(count_majority))
        return predictions

