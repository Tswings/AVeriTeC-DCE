import os
import json
from tqdm import tqdm
import html
import unicodedata
from nltk.tokenize import sent_tokenize, TreebankWordTokenizer
# ---------------------------------------------------------------------------
# # load simcse
from simcse import SimCSE
simcse_model = SimCSE("princeton-nlp/sup-simcse-roberta-base")
import sys
sys.path.append('presumm')
from presumm import train
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# decontextualization
from os import path
import tensorflow as tf
import tensorflow_text

MODEL_SIZE = "base"  # @param["base", "3B", "11B"]
DATASET_BUCKET = '../decontext/decontext_dataset'
SAVED_MODELS = {
    "base": f'{DATASET_BUCKET}/t5_base/1611267950',
    "3B": f'{DATASET_BUCKET}/t5_3B/1611333896',
    "11B": f'{DATASET_BUCKET}/t5_11B/1605298402'
}

SAVED_MODEL_PATH = SAVED_MODELS[MODEL_SIZE]
DEV = path.join(DATASET_BUCKET, 'decontext_dev.jsonl')
SAVED_MODEL_PATH = path.join(DATASET_BUCKET, 't5_base/1611267950')


def load_predict_fn(model_path):
    print("Loading SavedModel in eager mode.")
    imported = tf.saved_model.load(model_path, ["serve"])
    return lambda x: imported.signatures['serving_default'](
        tf.constant(x))['outputs'].numpy()


predict_fn = load_predict_fn(SAVED_MODEL_PATH)

def decontextualize(input):
    return predict_fn([input])[0].decode('utf-8')

# ---------------------------------------------------------------------------


def covert_ascii_to_char(sentences):
    # ascii_list = []
    tokenizer = TreebankWordTokenizer()
    ascii_dict = dict()
    sents_without_ascii = []
    for i, sent in enumerate(sentences):
        new_word = []
        sent = html.unescape(sent)

        if not sent.isascii():
            for word in tokenizer.tokenize(sent):
                if not word.isascii():
                    _word = unicodedata.normalize('NFD', word).encode('ascii', 'ignore').decode('utf-8')
                    if _word:
                        if not word.isalpha() or len(word) == len(_word):
                            ascii_dict[_word] = word
                            word = _word
                new_word.append(word)
            _sent = ' '.join(new_word)
            _sent = (_sent[:-2] + '.') if _sent[-2:] == ' .' else _sent
            sents_without_ascii.append(_sent)
        else:
            sents_without_ascii.append(sent)

    return sents_without_ascii, ascii_dict


def create_input(paragraph,
                 target_sentence_idx,
                 page_title='',
                 section_title=''):
    """Creates a single Decontextualization example input for T5.

    Args:
      paragraph: List of strings. Each string is a single sentence.
      target_sentence_idx: Integer index into `paragraph` indicating which
        sentence should be decontextualized.
      page_title: Optional title string. Usually Wikipedia page title.
      section_title: Optional title of section within page.
    """
    prefix = ' '.join(paragraph[:target_sentence_idx])
    target = paragraph[target_sentence_idx]
    suffix = ' '.join(paragraph[target_sentence_idx + 1:])
    return ' [SEP] '.join((page_title, section_title, prefix, target, suffix))


def run_decontextualization_in_qa(sentences, ori_text):
    sentences.append(ori_text)
    idx = len(sentences) - 1

    sents_without_ascii, ascii_dict = covert_ascii_to_char(sentences)
    dec_sent = decontextualize(create_input(sents_without_ascii, idx, '', ''))
    #
    n_fea, n_infea, n_unnec = 0, 0, 0
    if "DONE ####" in dec_sent:
        n_fea = 1
    elif "IMPOSSIBLE ####" in dec_sent:
        n_infea = 1
    else:       #if "UNNECESSARY ####" in dec_sent:
        n_unnec = 1
    #

    assert n_fea + n_infea + n_unnec == 1

    dec_sent_list = dec_sent.split('####')
    if 'DONE' in dec_sent:
        dec_sent_text = dec_sent_list[1].strip()
        if dec_sent_text:
            while dec_sent_text[0] == '"' and dec_sent_text[-1] == '"':
                dec_sent_text = dec_sent_text[1:-1]

        if '&amp;' in dec_sent_text:
            dec_sent_text = dec_sent_text.replace('&amp;', "&")
        if ' ⁇ ""' in dec_sent_text:
            dec_sent_text = dec_sent_text.replace(' ⁇ ""', '"')
        if '⁇' in dec_sent_text:
            dec_sent_text = sentences[idx]
            return dec_sent_text, n_fea, n_infea, n_unnec
        for key in ascii_dict.keys():
            if key in dec_sent_text and key not in sentences[idx]:
                dec_sent_text = dec_sent_text.replace(key, ascii_dict[key])

        simcse_score = simcse_model.similarity(sentences[idx], dec_sent_text)
        if simcse_score <= 0.6:  # best para: 0.6. 1) 0.8: 0.350; 2) 0.4: 0.351:
            return sentences[idx], n_fea, n_infea, n_unnec
        decontext_sent = dec_sent_text
    else:
        decontext_sent = sentences[idx]

    return decontext_sent, n_fea, n_infea, n_unnec


def sentence_decontextualisation(input_path):
    #
    samples = [json.loads(l) for l in open(input_path, 'r').readlines()]

    for ids, sample in tqdm(enumerate(samples)):
        ranked_sents_id = sample['sents_order_by_bertsum']
        ranked_sents = [(sent_id, sample['sentences'][sent_id]) for sent_id in ranked_sents_id]
        qa_pairs_list = sample['final_claims_cw']
        dec_sentences = []
        for idx, (sent_id, ori_text) in enumerate(ranked_sents):
            qa_pairs = qa_pairs_list[idx]

            dec_text = ori_text
            if sent_id != 0:  # Don't decontextualise the first sentence
                qa_as_context = []
                for qa in qa_pairs:
                    qa_as_context.append(qa[0])
                #
                dec_text, n_fea, n_infea, n_unnec = run_decontextualization_in_qa(qa_as_context, ori_text)
            #
            dec_sentences.append(dec_text)

        sample['dec_sentences'] = dec_sentences

    return samples


def main():
    #
    data_path = "all_data"

    # 3.1 decontextualisation
    input_path = "{}/7_highquality_context.json".format(data_path)
    output_path = "{}/8_decontextualised_claim.jsonl".format(data_path)

    if not os.path.exists(output_path):
        samples = sentence_decontextualisation(input_path)

        # save
        with open(output_path, 'w') as f:
            for sample in samples:
                f.write(json.dumps(sample) + '\n')
    else:
        samples = [json.loads(l) for l in open(output_path, 'r').readlines()]


    # 3.2 rerank decontextualised sentences
    rerank_path = "{}/9_dec_claim_for_rerank.jsonl".format(data_path)
    output_path = "{}/9_reranked_decontextualised_claim.jsonl".format(data_path)

    if not os.path.exists(rerank_path):
        for sample in tqdm(samples):
            sample['fulltext_ori'] = sample['fulltext']
            cand_claim = []
            for cc in sample['dec_sentences']:
                cand_claim.append(cc)

            sample['fulltext'] = ' '.join(cand_claim)

        with open(rerank_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False)

    #
    if not os.path.exists(output_path):
        sent_with_score = train.main()

        for idx, sample in enumerate(samples):
            print('id={}/{}'.format(idx, len(samples)))
            claims_selected_id = sent_with_score[idx][2]
            claims_selected_id_ordered = sent_with_score[idx][3][0].tolist()
            claims_id_order_by_bertsum = [claims_selected_id[pos] for pos in claims_selected_id_ordered]
            claims_order_by_bertsum = [sent_with_score[idx][0][pos] for pos in claims_selected_id_ordered]

            sample['claims_id_order_by_bertsum'] = claims_id_order_by_bertsum
            sample['claims_order_by_bertsum'] = claims_order_by_bertsum

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(samples, f, ensure_ascii=False)
    else:
        print("reranked decontextualised sentences have finished.")



if __name__ == "__main__":
    main()