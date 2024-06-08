#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by zd302 at 07/06/2024
import os
import json
import spacy
import stanza
import numpy as np
from tqdm import tqdm
from nltk.translate.chrf_score import sentence_chrf
import sys
sys.path.append('presumm')
from presumm import train
from nltk.tokenize import sent_tokenize
#
import torch
import torch.nn as nn
from simcse import SimCSE
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration
from transformers import AutoModelForSequenceClassification
from transformers.models.roberta.modeling_roberta import RobertaModel
#


# step 2
def sentence_ranking_by_BertSum(all_avail_url_files):

    # load configs
    configs = dict()
    configs['task'] = 'ext'
    configs['mode'] = 'test_text'
    configs['test_from'] = 'presumm/save_model/bertext_cnndm_transformer.pt'
    configs['text_src'] = 'all_data/1_all_available_url_fulltext.json'
    configs['result_path'] = 'presumm/results/ootb_output'
    configs['alpha'] = 0.95
    configs['log_file'] = 'presumm/logs/test.log'
    configs['visible_gpus'] = '0'

    # use BertSum to select candidate central sentences
    sent_with_score = train.main(configs)

    samples = json.load(open(all_avail_url_files, 'r'))
    for idx, sample in enumerate(samples):
        # split the fulltext into sentences
        fulltext = sample['fulltext']
        if fulltext[0] in ["“", "'", "”"] and fulltext[-1] in ["“", "'", "”"]:
            fulltext = fulltext[1:-1]
        sentences = sent_tokenize(fulltext)

        sample['sents_id_selected_by_bertsum'] = sent_with_score[idx][2]
        sample['sents_selected_by_bertsum'] = sent_with_score[idx][0]
        sample['sents_with_scores_by_bertsum'] = sent_with_score[idx][1].tolist()

        sample['sents_order_by_bertsum'] = sent_with_score[idx][4]
        sample['sent_texts_order_by_bertsum'] = sent_with_score[idx][5]
        sample['sentences'] = sentences

    return samples


# step 3
def get_phrases(tree, label):
    if tree.is_leaf():
        return []
    results = []
    for child in tree.children:
        results += get_phrases(child, label)

    if tree.label == label:
        return [' '.join(tree.leaf_labels())] + results
    else:
        return results


def candidate_answer_extraction(samples):
    # load nlp tools
    nlp = spacy.load('en_core_web_lg')
    stanza_nlp = stanza.Pipeline(lang='en', processors='tokenize,pos,constituency')
    #
    fulltext_processed = dict()
    for sample in tqdm(samples, desc="Candidate Generation"):
        fulltext = sample['fulltext']
        #
        chrf_all_sents = [sentence_chrf(sample['claim'].split(), sent.split()) for sent in sample['sentences']]
        top3_sent_id_in_all_sents = np.argsort(-np.array(chrf_all_sents)).tolist()[:3]
        top3_chrf_in_all_sents = [chrf_all_sents[i] for i in top3_sent_id_in_all_sents]
        sample['top3_sent_in_all_sents'] = [top3_sent_id_in_all_sents, top3_chrf_in_all_sents]

        chrf_ext_sents = [sentence_chrf(sample['claim'].split(), sample['sentences'][_id].split()) for _id in
                          sample['sents_order_by_bertsum']]
        order_top3_sent_id_in_ext_sents = np.argsort(-np.array(chrf_ext_sents)).tolist()[:3]
        top3_sent_id_in_ext_sents = [sample['sents_order_by_bertsum'][j] for j in order_top3_sent_id_in_ext_sents]
        top3_chrf_in_ext_sents = [chrf_ext_sents[i] for i in order_top3_sent_id_in_ext_sents]
        sample['top3_sent_in_ext_sents'] = [top3_sent_id_in_ext_sents, top3_chrf_in_ext_sents]

        # extract candidate answers
        if fulltext not in fulltext_processed.keys():
            sample['candidate_answers'] = []

            central_sents_sel_by_bertsum = [i for i in sample['sents_order_by_bertsum']]
            candidate_central_sentences = [sample['sentences'][i] for i in central_sents_sel_by_bertsum]

            # select entities from candidate sentences
            candidate_answers = []
            for sent in candidate_central_sentences:
                if sent:
                    candidate_answers_list = []
                    doc = nlp(sent)
                    stanza_doc = stanza_nlp(sent)

                    # extract entities from sents
                    ents = [ent.text for sent in doc.sents for ent in sent.noun_chunks]
                    ents += [ent.text for sent in doc.sents for ent in sent.ents]
                    ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'NP')]
                    #
                    ents += [phrase for sent in stanza_doc.sentences for phrase in get_phrases(sent.constituency, 'VP')]
                    ents += [word.text for sent in stanza_doc.sentences for word in sent.words if
                             word.upos in ['VERB', 'ADV', 'ADJ', 'NOUN']]

                    # extract negation from sents
                    negations = [word for word in ['not', 'never'] if word in sample['fulltext']]
                    candidate_answers_list.extend(list(set(ents + negations)))
                    #
                    candidate_answers.append(candidate_answers_list)

            sample['candidate_answers'].extend(candidate_answers)
            fulltext_processed[fulltext] = sample['candidate_answers']
        else:
            print("Load the processed fulltext")
            sample['candidate_answers'] = fulltext_processed[fulltext]

    return samples


# step 4
def question_generation(samples):
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained('Salesforce/mixqg-base').to(device)

    batch_size = 10

    def format_inputs(context: str, answer: str):
        # return f"answer:{answer} context:{context}"
        return f"{answer} \\n {context}"

    fulltext_processed = dict()  # Load the processed data if the fulltext is duplicated
    for sample in tqdm(samples, desc="Generating Questions"):
        fulltext = sample['fulltext']

        if fulltext not in fulltext_processed.keys():
            sample['generated_question'] = []

            cand_sents = [i for i in sample['sents_order_by_bertsum']]  # sentences
            assert len(cand_sents) == len(sample['candidate_answers'])

            for idx in range(len(sample['candidate_answers'])):
                # print("idx:{}/{}".format(idx, len(sample['candidate_answers'])))
                texts = []
                _sentence = sample['sentences'][cand_sents[idx]]
                _candidate_ansewrs = sample['candidate_answers'][idx]

                for cand_ans in _candidate_ansewrs:
                    texts.append(format_inputs(_sentence, cand_ans))

                gen_question = []
                if texts:
                    for idy in range(0, len(texts), batch_size):
                        input_ids = tokenizer(texts[idy:idy + batch_size], return_tensors="pt", padding='longest',
                                              truncation=True, max_length=1024).input_ids.to(device)
                        generated_ids = model.generate(input_ids, max_length=32, num_beams=4)
                        output = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        gen_question.extend(output)
                    sample['generated_question'].append(gen_question)
                else:
                    sample['generated_question'].append(gen_question)

            fulltext_processed[fulltext] = sample['generated_question']
        else:
            print("Load the processed fulltext")
            sample['generated_question'] = fulltext_processed[fulltext]

    return samples


# step 5
def qa_generation(samples):
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load model
    model_name = 'allenai/unifiedqa-v2-t5-base-1251000'
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = T5ForConditionalGeneration.from_pretrained(model_name).to(device)

    #
    fulltext_processed = dict()  # Load the processed data if the fulltext is duplicated
    for sample in tqdm(samples, desc="Generating Answers"):
        fulltext = sample['fulltext']
        cand_sents = [i for i in sample['sents_order_by_bertsum']]  # sentences

        if fulltext not in fulltext_processed.keys():
            sample['answer'] = []

            for idx, questions in enumerate(sample['generated_question']):
                print("id={}/{}".format(idx, len(sample['generated_question'])))

                # use the fulltext as context to answer the question; if the fulltext is too long, use the (head-10, tail-10) of the sentence as context
                if len(fulltext) <= 400:
                    context = fulltext
                else:
                    if cand_sents[idx] >= 10:
                        context = sample['sentences'][(cand_sents[idx] - 10):(cand_sents[idx] + 10)]
                    else:
                        context = sample['sentences'][0:(cand_sents[idx] + 10)]

                current_answers = []
                question_processed = dict()  # load it if the question is already answered
                for idy, question in enumerate(questions):
                    if question not in question_processed.keys():
                        input_ids = tokenizer.encode(f"{question} \n {context}", return_tensors='pt').to(device)

                        with torch.no_grad():
                            outputs = model.generate(input_ids, num_beams=4, do_sample=False)
                            predict_answer_tokens_string = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]

                        current_answers.append(predict_answer_tokens_string.strip())
                        question_processed[question] = predict_answer_tokens_string.strip()
                    else:
                        print("Load the processed question")
                        current_answers.append(question_processed[question])
                #
                sample['answer'].append(current_answers)

            fulltext_processed[fulltext] = sample['answer']
        else:
            print("Load the processed fulltext")
            sample['answer'] = fulltext_processed[fulltext]

    return samples


# step 6
def qa_to_context(samples):
    #
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #
    pretrain_model_path = "qa2claim_model"
    tokenizer = AutoTokenizer.from_pretrained('t5-base')
    model = AutoModelForSeq2SeqLM.from_pretrained(pretrain_model_path).to(device)

    def format_inputs(question: str, answer: str):
        return f"{answer} \\n {question}"

    #
    fulltext_processed = dict()     # Load the processed data if the fulltext is duplicated
    for sample in tqdm(samples, desc="Converting QA to statements"):
        fulltext = sample['fulltext']

        if fulltext not in fulltext_processed.keys():
            generated_questions = sample['generated_question']
            generated_answers = sample['answer']

            sample['candidate_claims'] = []
            for questions, answers in zip(generated_questions, generated_answers):
                candidate_corrections_list = []
                qa_pair_processed = dict()      # load it if two qa-pairs are the same

                for idx, answer in enumerate(answers):
                    input_text = format_inputs(questions[idx], answer)

                    if input_text not in qa_pair_processed.keys():
                        input_text = format_inputs(questions[idx], answer)
                        input_ids = tokenizer(input_text, return_tensors="pt", padding='longest', truncation=True,
                                              max_length=512).input_ids.to(device)

                        generated_ids = model.generate(input_ids, max_length=64, num_beams=4, early_stopping=True)
                        candidate_corrections = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                        candidate_corrections_list.append(candidate_corrections)

                        qa_pair_processed[input_text] = candidate_corrections
                    else:
                        # load
                        print("Load the processed qa pair")
                        candidate_corrections_list.append(qa_pair_processed[input_text])

                sample['candidate_claims'].append(candidate_corrections_list)

            fulltext_processed[fulltext] = sample['candidate_claims']
        else:
            print("Load the processed fulltext")
            sample['candidate_claims'] = fulltext_processed[fulltext]

    return samples


# step 7
def gen_highquality_context(samples):
    #
    simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-large")
    cw_labels = ['Non-Factual Statement(NFS)', 'Unimportant Factual Statement(UFS)', 'Check-worthy Factual Statement(CFS)']
    cw_tokenizer = AutoTokenizer.from_pretrained("decontext/claimbuster_model", use_auth_token=True)
    cw_model = AutoModelForSequenceClassification.from_pretrained("decontext/claimbuster_model", use_auth_token=True)

    #
    class RobertaForSequenceClassification(nn.Module):
        def __init__(self, tagset_size):
            super(RobertaForSequenceClassification, self).__init__()
            self.tagset_size = tagset_size

            self.roberta_single = RobertaModel.from_pretrained(pretrain_model_dir)
            self.single_hidden2tag = RobertaClassificationHead(bert_hidden_dim, tagset_size)

        def forward(self, input_ids, input_mask):
            outputs_single = self.roberta_single(input_ids, input_mask, None)
            hidden_states_single = outputs_single[1]

            score_single = self.single_hidden2tag(hidden_states_single)  # (batch, tag_set)
            return score_single

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    bert_hidden_dim = 1024
    pretrain_model_dir = 'roberta-large'

    label_list = ["entailment", "not_entailment"]  # , "contradiction"]
    num_labels = len(label_list)

    class RobertaClassificationHead(nn.Module):
        """wenpeng overwrite it so to accept matrix as input"""

        def __init__(self, bert_hidden_dim, num_labels):
            super(RobertaClassificationHead, self).__init__()
            self.dense = nn.Linear(bert_hidden_dim, bert_hidden_dim)
            self.dropout = nn.Dropout(0.1)
            self.out_proj = nn.Linear(bert_hidden_dim, num_labels)

        def forward(self, features):
            x = features  # [:, 0, :]  # take <s> token (equiv. to [CLS])
            x = self.dropout(x)
            x = self.dense(x)
            x = torch.tanh(x)
            x = self.dropout(x)
            x = self.out_proj(x)
            return x

    # load model
    model = RobertaForSequenceClassification(num_labels).to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrain_model_dir)
    checkpoint = torch.load('docnli_model/DocNLI.pretrained.RoBERTA.model.pt')
    model.load_state_dict(checkpoint)

    def entailment_score(text1, text2):
        encoded_ctx = tokenizer.encode(text1)[:-1]          # remove [SEP]
        encoded_correction = tokenizer.encode(text2)[1:]    # remove [CLS]

        encoded_ctx_truncated = encoded_ctx[:512 - 1 - len(encoded_correction)]  # - [SEP] - encoded_correction
        input_ids = torch.LongTensor(encoded_ctx_truncated + [tokenizer.sep_token_id] + encoded_correction).unsqueeze(
            0).to(device)
        attention_mask = torch.LongTensor([1] * len(input_ids)).unsqueeze(0).to(device)
        inputs = {'input_ids': input_ids, 'input_mask': attention_mask}

        with torch.no_grad():
            model.eval()
            logits = model(**inputs)
            probs = torch.nn.Softmax(dim=1)(logits)
            correct_prob = probs[0][0].item()

        return correct_prob

    fulltext_processed = dict()  # Load the processed data if the fulltext is duplicated
    for sample in tqdm(samples, desc="Running DocNLI"):
        fulltext = sample['fulltext']
        cand_sents = [i for i in sample['sents_order_by_bertsum']]

        if fulltext not in fulltext_processed.keys():
            sample['claim_rank_by_entail_score'] = []
            sample['claim_rank_by_simcse_score'] = []
            sample['final_claims_cw'] = []
            sample['final_claims'] = []

            for idx, gen_claim in enumerate(sample['candidate_claims']):
                cand_claim_by_entail = []
                cand_claim_by_simcse = []
                cand_claim_rmv_dup = []             # Remove duplicate claim

                if gen_claim:
                    gen_claim_entail_scores = []    # entailment score between 'sentence' and 'generated_claim'
                    gen_claim_simcse_scores = []
                    claim_processed = dict()        # load it if the claim is processed

                    for idy, _claim in enumerate(gen_claim):
                        if _claim[0] not in claim_processed.keys():
                            correct_prob = entailment_score(sample['sentences'][cand_sents[idx]], _claim[0])
                            # chrf_score = sentence_chrf(sample['sentences'][cand_sents[idx]].split(), _claim[0].split())
                            gen_claim_entail_scores.append(correct_prob)
                            claim_processed[_claim[0]] = correct_prob
                        else:
                            print("load the processed claim")
                            gen_claim_entail_scores.append(claim_processed[_claim[0]])

                    # select top-k claims, generated claims should be entailed in the sentence
                    topk_claims_id = range(len(gen_claim_entail_scores))

                    if not topk_claims_id:
                        topk_claims_id = np.argsort(-np.array(gen_claim_entail_scores)).tolist()[:5]  #

                    for idz, _claim in enumerate(gen_claim):
                        if idz in topk_claims_id:
                            if [_claim[0], gen_claim_entail_scores[idz]] not in cand_claim_by_entail:  # [claim, chrf]
                                cand_claim_by_entail.append([_claim[0], gen_claim_entail_scores[idz]])

                    # claims ranked by entailment_score
                    cand_claim_by_entail = sorted(cand_claim_by_entail, key=lambda x: x[1], reverse=True)

                    # claims ranked by simcse_score between the sentence and claim
                    if cand_claim_by_entail:
                        cand_claim_by_entail_list = [c for c, s in cand_claim_by_entail]
                        # simcse_score = simcse_model.similarity(sample['sentences'][cand_sents[idx]], cand_claim_by_entail_list)
                        simcse_score = (simcse_model.similarity(sample['sentences'][cand_sents[idx]],
                                                                cand_claim_by_entail_list)).ravel().tolist()
                        gen_claim_simcse_scores.extend(simcse_score)
                        cand_claim_by_simcse_id = np.argsort(-np.array(gen_claim_simcse_scores)).tolist()
                        cand_claim_by_simcse.extend(
                            [cand_claim_by_entail_list[i], gen_claim_simcse_scores[i]] for i in cand_claim_by_simcse_id)

                    sample['claim_rank_by_entail_score'].append(cand_claim_by_entail)
                    sample['claim_rank_by_simcse_score'].append(cand_claim_by_simcse)

                    # filter claim_2, if claim_2 is entailed in claim_1
                    cand_claim_rmv_dup.extend([c for c, s in cand_claim_by_simcse])

                    filter_by_entail_ids = []
                    for i in reversed(range(len(cand_claim_rmv_dup))):
                        for j in reversed(range(i)):
                            if i not in filter_by_entail_ids:
                                entail_prob = entailment_score(cand_claim_rmv_dup[j], cand_claim_rmv_dup[i])
                                if entail_prob > 0.9:
                                    filter_by_entail_ids.append(i)

                    for i in filter_by_entail_ids:
                        del cand_claim_rmv_dup[i]

                    # filter claim 2 if claim 2 is highly related to claim 1
                    # if cand_claim_rmv_dup:
                    simcse_sents = simcse_model.similarity(cand_claim_rmv_dup, cand_claim_rmv_dup)
                    filter_by_simcse_ids = []
                    for i in reversed(range(len(cand_claim_rmv_dup))):
                        for j in reversed(range(i)):
                            if i not in filter_by_simcse_ids:
                                # a = simcse_sents[i][j]
                                if simcse_sents[i][j] > 0.85:  # 0.8
                                    filter_by_simcse_ids.append(i)

                    for i in filter_by_simcse_ids:
                        del cand_claim_rmv_dup[i]

                    # check-worthy classification
                    cand_claim_with_check_worthy = []
                    cand_claim_final = []
                    for claim_text in cand_claim_rmv_dup:
                        cw_sent_inputs = cw_tokenizer(claim_text, return_tensors="pt")
                        cw_sent_outputs = cw_model(**cw_sent_inputs)
                        sent_logits = cw_sent_outputs.logits.tolist()[0]
                        cw_sent_class = np.argmax(sent_logits)
                        sent_label = cw_labels[int(cw_sent_class)]
                        cand_claim_with_check_worthy.append([claim_text, sent_label, sent_logits])
                        #
                        if sent_label in ['Unimportant Factual Statement(UFS)', 'Check-worthy Factual Statement(CFS)']:
                            cand_claim_final.append([claim_text])

                    sample['final_claims_cw'].append(cand_claim_with_check_worthy)
                    sample['final_claims'].append(cand_claim_final)
                else:
                    sample['claim_rank_by_entail_score'].append(cand_claim_by_entail)
                    sample['claim_rank_by_simcse_score'].append(cand_claim_by_simcse)
                    sample['final_claims_cw'].append([])
                    sample['final_claims'].append([])

                fulltext_processed[fulltext] = [sample['claim_rank_by_entail_score'],
                                                sample['claim_rank_by_simcse_score'], sample['final_claims_cw'],
                                                sample['final_claims']]
        else:
            print("Load the processed fulltext")
            sample['claim_rank_by_entail_score'], sample['claim_rank_by_simcse_score'], sample['final_claims_cw'], \
            sample['final_claims'] = fulltext_processed[fulltext]

    return samples


def main():
    data_path = "all_data"

    # ---------------------------------------------------------------------------
    # step 1: extracts URLs available for claim extraction and corresponding text data from AVeriTeC:
    all_avail_url_files = "{}/1_all_available_url_fulltext.json".format(data_path)
    if not os.path.exists(all_avail_url_files):
        print("***** run python 1_extract_texts_from_url.py *****")

    # ---------------------------------------------------------------------------
    # step 2: extracts candidate central sentences from text data
    ranked_sentence_file = "{}/2_sent_ranked_by_bertsum1.json".format(data_path)
    if not os.path.exists(ranked_sentence_file):
        ranked_samples = sentence_ranking_by_BertSum(all_avail_url_files)
        # save
        with open(ranked_sentence_file, 'w') as f:
            for sample in ranked_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        ranked_samples = json.load(open(ranked_sentence_file, 'r'))

    # ---------------------------------------------------------------------------
    # step 3: extracts candidates answers from candidate central sentences:
    cand_ans_extraction_file = "{}/3_generated_candidates.jsonl".format(data_path)
    if not os.path.exists(cand_ans_extraction_file):
        cand_ans_samples = candidate_answer_extraction(ranked_samples)
        # save
        with open(cand_ans_extraction_file, 'w') as f:
            for sample in cand_ans_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        cand_ans_samples = [json.loads(l) for l in open(cand_ans_extraction_file, 'r').readlines()]

    # ---------------------------------------------------------------------------
    # step 4: generates questions for each candidates answer
    gen_question_file = "{}/4_generated_questions.jsonl".format(data_path)
    if not os.path.exists(gen_question_file):
        gen_que_samples = question_generation(cand_ans_samples)
        # save
        with open(gen_question_file, 'w') as f:
            for sample in gen_que_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        gen_que_samples = [json.loads(l) for l in open(gen_question_file, 'r').readlines()]

    # ---------------------------------------------------------------------------
    # step 5: answers generated questions:
    gen_answer_file = "{}/5_generated_answers.jsonl".format(data_path)
    if not os.path.exists(gen_answer_file):
        gen_answer_samples = qa_generation(gen_que_samples)
        # save
        with open(gen_answer_file, 'w') as f:
            for sample in gen_answer_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        gen_answer_samples = [json.loads(l) for l in open(gen_answer_file, 'r').readlines()]

    # ---------------------------------------------------------------------------
    # step 6: converts qa pairs to declarative sentences:
    gen_context_file = "{}/6_generated_context.jsonl".format(data_path)
    if not os.path.exists(gen_context_file):
        gen_context_samples = qa_to_context(gen_answer_samples)
        # save
        with open(gen_context_file, 'w') as f:
            for sample in gen_context_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        gen_context_samples = [json.loads(l) for l in open(gen_context_file, 'r').readlines()]

    # ---------------------------------------------------------------------------
    # step 7: removes redundant sentences from generated contexts:
    gen_context_file = "{}/7_highquality_context.jsonl".format(data_path)
    if not os.path.exists(gen_context_file):
        gen_highcontext_samples = gen_highquality_context(gen_context_samples)
        # save
        with open(gen_context_file, 'w') as f:
            for sample in gen_highcontext_samples:
                f.write(json.dumps(sample) + '\n')
    else:
        print("No high-context file!")
        # gen_highcontext_samples = json.load(open(gen_context_file, 'r'))


    print("hello")


if __name__ == "__main__":
    main()
