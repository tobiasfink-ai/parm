import os
import csv
import argparse
import json
import jsonlines
import random
import numpy as np
from pyserini.search import SimpleSearcher
from preprocessing.stat_corpus import count_words
import hydra
from omegaconf import DictConfig
random.seed(42)
from tqdm import tqdm
from pathlib import Path


def read_in_samples_task2(case_corpus, labels, mode):
    samples = []
    
    for case_id, rel_ids in tqdm(labels.items(), desc=f"Mode: {mode}"):
        positive_ctxs = []
        positive_ids = set()
        case_data = case_corpus[case_id]
        for rel_id in rel_ids:
            positive_ids.add(rel_id)
            doc_rel_text = case_data["paragraph_texts"][rel_id]
            ctx = {
                "title": "",
                "text": doc_rel_text,
                #"score": 0,
                "case_id": case_id,
                "passage_num": Path(rel_id).stem,
                "psg_id": f"{case_id}_{Path(rel_id).stem}"
            }
            positive_ctxs.append(ctx)

        negative_candidate_ids = list(p_id for p_id in case_data["paragraph_texts"].keys() if p_id not in positive_ids)
        negative_sample_k = len(positive_ids)
        if negative_sample_k > len(negative_candidate_ids):
            print(f"Small negative sample in case {case_id} with rel_ids {rel_ids}")
            negative_sample_k = len(negative_candidate_ids)
        random_negative_ids = random.sample(negative_candidate_ids, k=negative_sample_k)

        hard_negative_ctxs = []
        for irrel_id in random_negative_ids:
            doc_rel_text = case_data["paragraph_texts"][irrel_id]
            ctx = {
                "title": "",
                "text": doc_rel_text,
                #"score": 0,
                "case_id": case_id,
                "passage_num": Path(irrel_id).stem,
                "psg_id": f"{case_id}_{Path(irrel_id).stem}"
            }
            hard_negative_ctxs.append(ctx)
        
        sample = {
            "question": case_data["query_text"],
            "answers": [],
            "positive_ctxs": positive_ctxs,
            "negative_ctxs": [],
            "hard_negative_ctxs": hard_negative_ctxs
        }

        samples.append(sample)

    return samples


def jsonl_per_paragraph(train_dir: str):
    list_dir = [x for x in os.walk(train_dir)]

    for sub_dir in list_dir[0][1]:
        print(sub_dir)
        with jsonlines.open(os.path.join(train_dir, sub_dir, 'candidates.jsonl'), mode='w') as writer:
            # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
            list_sub_dir_paragraphs = [x for x in os.walk(os.path.join(train_dir, sub_dir, 'paragraphs'))]
            for paragraph in list_sub_dir_paragraphs[0][2]:
                with open(os.path.join(train_dir, sub_dir, 'paragraphs', paragraph), 'r') as paragraph_file:
                    para_text = paragraph_file.read().splitlines()[1:]
                    writer.write({'id': '{}_{}'.format(sub_dir, paragraph.split('.')[0]),
                                  'contents': ' '.join([text.strip().replace('\n', '') for text in para_text]).strip()})


def search_indices_per_paragraph(train_dir: str):
    list_dir = [x for x in os.walk(train_dir)]

    with open(os.path.join(train_dir, 'failed_dirs.txt'), 'w') as failed_dir:
        for sub_dir in list_dir[0][1]:
            print(sub_dir)
            try:
                # read in query text
                with open(os.path.join(train_dir, sub_dir, 'entailed_fragment.txt'), 'r') as entailed_fragment:
                    query_text_lines = entailed_fragment.read().splitlines()
                    query_text = ' '.join([text.strip().replace('\n', '') for text in query_text_lines]).strip()

                searcher = SimpleSearcher(os.path.join(train_dir, sub_dir, 'index'))
                searcher.set_bm25(0.9, 0.4)

                hits = searcher.search(query_text, 200)

                # Print the first 50 hits:
                with open(os.path.join(train_dir, sub_dir, 'bm25_top200.txt'), "w", encoding="utf8") as out_file:
                    for hit in hits:
                        out_file.write(f'{hit.docid:55} {hit.score:.5f}\n')
                        #print(f'{hit.docid:55} {hit.score:.5f}\n')
            except:
                print('failed for {}'.format(sub_dir))
                failed_dir.write('{}\n'.format(sub_dir))


def analyze_samples(samples: list, output_dir=None):
    no_rel = []
    len_para = []
    len_quest = []
    for sample in samples:
        no_rel.append(len(sample.get('positive_ctxs')))
        len_para.extend([count_words(pos_para.get('text')) for pos_para in sample.get('positive_ctxs')])
        len_para.extend([count_words(pos_para.get('text')) for pos_para in sample.get('hard_negative_ctxs')])
        len_quest.append(count_words(sample.get('question')))

    print('Average number of relevant passages per query is {}'.format(np.mean(no_rel)))
    print('Average length of passages is {}'.format(np.mean(len_para)))
    print('Average length of queries is {}'.format(np.mean(len_quest)))

    if output_dir:
        with open(os.path.join(output_dir, 'analysis.txt'), 'w') as out:
            out.write('Average number of relevant passages per query is {}\n'.format(np.mean(no_rel)))
            out.write('Average length of passages is {}\n'.format(np.mean(len_para)))
            out.write('Average length of queries is {}\n'.format(np.mean(len_quest)))


def write_to_json(samples: list, output_dir: str):
    with open(os.path.join(output_dir, 'samples.json'),'w') as out:
        json.dump(samples, out)


def get_label_split(cfg):
    """
    Since COLIEE 2024 has not provided test set labels, split here from official training set
    """
    label_file = cfg.task2.label_file_train_original
    label_train_val_split = cfg.task2.label_train_val_split
    if cfg.task2.label_train_test_split:
        label_train_test_split = cfg.task2.label_train_test_split
    else:
        label_train_test_split = 0

    # first preprocess label file
    with open(label_file, "r") as fp:
        label_dict = json.load(fp)
    
    # remove file extension from end of id
    for key in label_dict.keys():
        raw_ids = label_dict[key]
        label_dict[key] = [raw_id.split('.')[0] for raw_id in raw_ids]

    label_keys_list = list(label_dict.keys())
    total_labels = len(label_keys_list)
    labels_train = {} # first part
    labels_val = {} # second part
    labels_test = {} # third part
    for idx, label in enumerate(label_keys_list):
        if idx < (total_labels-label_train_val_split-label_train_test_split):
            labels_train[label] = label_dict[label]
        elif idx < (total_labels-label_train_test_split):
            labels_val[label] = label_dict[label]
        else:
            labels_test[label] = label_dict[label]
    

    return labels_train, labels_val, labels_test


def read_corpus(corpus_dir):
    """
    Read in corpus from COLIEE task 2. Each case has a query text and paragraph texts of the cited case
    """
    task2_cases = os.listdir(corpus_dir)
    case_corpus = {}
    for case_id in tqdm(task2_cases, desc="Read Corpus"):
        # read in query text
        case_folder = os.path.join(corpus_dir, case_id)
        if not os.path.isdir(case_folder):
            continue
        with open(os.path.join(corpus_dir, case_id, 'entailed_fragment.txt'), 'r') as entailed_fragment:
            query_text_lines = [line.strip().replace('\n', '') for line in entailed_fragment]
            query_text = ' '.join(query_text_lines).strip()
        
        # read in all paragraphs with their names and then choose the relevant ones and sample irrelevant ones!
        list_sub_dir_paragraphs = os.listdir(os.path.join(case_folder, 'paragraphs'))
        paragraph_texts = {}
        for paragraph in list_sub_dir_paragraphs:
            paragraph_num = Path(paragraph).stem
            with open(os.path.join(corpus_dir, case_id, 'paragraphs', paragraph), 'r') as paragraph_file:
                para_lines = [line.strip().replace('\n', '') for line in paragraph_file][1:] # skip first line, which is just the number
                para_text = ' '.join(para_lines).strip()
                paragraph_texts[paragraph_num] = para_text
        
        case_corpus[case_id] = {
            "query_text": query_text,
            "paragraph_texts": paragraph_texts
        }
    return case_corpus


@hydra.main(version_base=None, config_path="../config", config_name=None)
def main(cfg: DictConfig):
    corpus_dir = cfg.task2.corpus_dir
    output_dir = cfg.task2.output_dir

    datasets = {
        "train":{},
        "val":{}
    }
    if cfg.task2.label_train_test_split:
        datasets["test"] = {}

    for key in datasets.keys():
        datasets[key]["output_dir"] = os.path.join(output_dir, key)
    

    labels_train, labels_val, labels_test = get_label_split(cfg)

    label_file_train = cfg.task2.label_file_train
    label_file_val = cfg.task2.label_file_val

    datasets["train"]["labels"] = labels_train
    datasets["val"]["labels"] = labels_val

    with open(label_file_train, "w") as fp:
        json.dump(labels_train, fp, indent=4)
    with open(label_file_val, "w") as fp:
        json.dump(labels_val, fp, indent=4)

    if cfg.task2.label_train_test_split:
        datasets["test"] = {}
        label_file_test = cfg.task2.label_file_test
        datasets["test"]["labels"] = labels_test
        with open(label_file_test, "w") as fp:
            json.dump(labels_test, fp, indent=4)

    case_corpus = read_corpus(corpus_dir)
    

    #jsonl_per_paragraph(test_dir2)
    #search_indices_per_paragraph(train_dir)

    for mode, dataset in datasets.items():
        qrels = dataset["labels"]
        samples = read_in_samples_task2(case_corpus, qrels, mode=mode)
        analyze_samples(samples, dataset["output_dir"])
        write_to_json(samples, dataset["output_dir"])


if __name__ == "__main__":
    main()

