import os
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import json
import re
import pickle
import hydra
from omegaconf import DictConfig
from preprocessing.stat_corpus import lines_to_paragraphs, count_doc, only_string_in_dict, analyze_corpus_in_numbers, analyze_text_passages
from preprocessing.jsonlines_for_bm25_pyserini import jsonl_index_whole_doc, jsonl_index_doc_only_para, jsonl_index_para_separately


def only_english(paragraphs: dict):
    # check intro where the english version is, is it the first one or the second one of the paragraphs?
    # but only if there are multiple options for the paratgraph
    freen = '[English language version follows French language version]'
    freen2 = '[La version anglaise vient à la suite de la version française]'
    enfre = '[French language version follows English language version]'
    enfre2 = '[La version française vient à la suite de la version anglaise]'

    if enfre in paragraphs.get('intro')[0] or enfre2 in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[0]]})
    elif freen in paragraphs.get('intro')[0] or freen2 in paragraphs.get('intro')[0]:
        for key, value in paragraphs.items():
            if len(value) > 1:
                paragraphs.update({key: [value[1]]})
    return paragraphs


def read_in_para_lengths(corpus_dir: str, output_dir: str):
    '''
    reads in all files, separates them in intro, summary and the single paragraphs, counts the lengths
    only considers the english versions of the files, prints if it fails to read a certain file and
    stores it in failed_files
    :param corpus_dir: directory of the corpus containing the text files
    :param output_dir: output directory where the pickled files of the lengths of each file, the paragraphs of each
    file and the failed_files are stored
    :return: the lengths of each file, the paragraphs of each file and the failed_files
    '''
    lengths = {}
    dict_paragraphs = {}
    failed_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
            # file = '000028.txt'
            with open(os.path.join(corpus_dir, file), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip('\n') != ' ' and line.strip() != '']
                # remove fragment supressed and \xa0
                lines = [line.replace('<FRAGMENT_SUPPRESSED>', '').strip() for line in lines if
                         line.replace('<FRAGMENT_SUPPRESSED>', '').strip() != '']
                lines = [line.replace('\xa0', '').strip() for line in lines if
                         line.replace('\xa0', '').strip() != '']
                # remove lines with only punctuation
                lines = [line for line in lines if re.sub(r'[^\w\s]', '', line) != '']
                paragraphs = lines_to_paragraphs(lines)
                if paragraphs:
                    paragraphs = only_english(paragraphs)
                    paragraphs = only_string_in_dict(paragraphs)
                    # now analyze the intro, the summary and the length of the paragraphs
                    no_intro, no_summ, lengths_para = count_doc(paragraphs)
                    lengths.update({file.split('.')[0]: {'intro': no_intro, 'summary': no_summ,
                                                         'lengths_paragraphs': lengths_para}})
                    dict_paragraphs.update({file.split('.')[0]: paragraphs})
                    # print('lengths for file {} done'.format(file))
                else:
                    print('reading in of file {} doesnt work'.format(file))
                    failed_files.append(file)

    with open(os.path.join(output_dir, 'corpus_lengths.pickle'), 'wb') as f:
        pickle.dump(lengths, f)
    with open(os.path.join(output_dir, 'corpus_paragraphs.pickle'), 'wb') as f:
        pickle.dump(dict_paragraphs, f)
    with open(os.path.join(output_dir, 'corpus_failed_files.pickle'), 'wb') as f:
        pickle.dump(failed_files, f)

    return lengths, dict_paragraphs, failed_files


def preprocess_label_file(label_file: str):
    with open(label_file, 'rb') as f:
        labels = json.load(f)

    label_dict = {}
    for key, value in labels.items():
        label_dict.update({key.split('.')[0]: [val.split('.')[0] for val in value]})

    return label_dict


def read_in_docs(corpus_dir: str, output_dir: str, pickle_dir: str, removal=True):
    '''
    reads in all files, separates them in intro, summary and the single paragraphs, removes non-informative
    intros and summaries, writes them into jsonlines file, prints if it fails to read a certain file and
    stores it in failed_files
    :param corpus_dir: directory of the corpus containing the text files
    :param output_dir: output directory where the pickled files of the non-informative intros and summaries are
    :param removal: if non-informative text should be removed in the intros and the summaries
    :return:
    '''
    with open(os.path.join(pickle_dir, 'intro_text_often.pkl'), 'rb') as f:
        intro_often = pickle.load(f)
    with open(os.path.join(pickle_dir, 'summ_text_often.pkl'), 'rb') as f:
        summ_often = pickle.load(f)

    dict_paragraphs = {}
    failed_files = []
    for root, dirs, files in os.walk(corpus_dir):
        for file in files:
        #file = '001_001.txt'
            with open(os.path.join(corpus_dir, file), 'r') as f:
                lines = f.readlines()
                lines = [line.strip() for line in lines if line.strip('\n') != ' ' and line.strip() != '']
                # remove fragment supressed and \xa0
                lines = [line.replace('<FRAGMENT_SUPPRESSED>', '').strip() for line in lines if
                         line.replace('<FRAGMENT_SUPPRESSED>', '').strip() != '']
                lines = [line.replace('\xa0', '').strip() for line in lines if
                         line.replace('\xa0', '').strip() != '']
                lines = [' '.join(line.split()) for line in lines]
                # remove lines with only punctuation
                lines = [line for line in lines if re.sub(r'[^\w\s]', '', line) != '']
                paragraphs = lines_to_paragraphs(lines)
                if paragraphs:
                    paragraphs = only_english(paragraphs)
                    paragraphs = only_string_in_dict(paragraphs)
                    if removal:
                        if paragraphs.get('intro') in intro_often:
                            paragraphs.update({'intro': None})
                        if paragraphs.get('Summary:') in summ_often:
                            paragraphs.update({'Summary:': None})
                    dict_paragraphs.update({file.split('.')[0]: paragraphs})
                else:
                    print('reading in of file {} doesnt work'.format(file))
                    failed_files.append(file)

    #with open(os.path.join(output_dir, 'paragraphs_jsonlines.pickle'), 'wb') as f:
    #    pickle.dump(dict_paragraphs, f)
    #with open(os.path.join(output_dir, 'failed_files_jsonlines.pickle'), 'wb') as f:
    #    pickle.dump(failed_files, f)

    return dict_paragraphs, failed_files


def split_train_in_train_and_val(label_file: str, label_file_train: str, label_file_val: str):
    # split labels train into train and val
    with open(label_file, 'rb') as f:
        labels = json.load(f)

    label_train_wo_val = {}
    keys_list = list(labels.keys())
    for i in range(550):
        label_train_wo_val.update({keys_list[i]:labels.get(keys_list[i])})

    label_val = {}
    for i in range(550, len(keys_list)):
        label_val.update({keys_list[i]: labels.get(keys_list[i])})

    with open(label_file_train, 'w') as f:
        json.dump(label_train_wo_val, f)
    with open(label_file_val, 'w') as f:
        json.dump(label_val, f)


@hydra.main(version_base=None, config_path="../config", config_name=None)
def main(cfg: DictConfig):
    mode = cfg.task1.mode
    corpus_dir = cfg.task1.corpus_dir
    output_dir = os.path.join(cfg.task1.output_dir, mode)
    pickle_dir = cfg.task1.pickle_dir
    
    label_file_train_original = cfg.task1.label_file_train_original
    label_file_train = cfg.task1.label_file_train
    label_file_val = cfg.task1.label_file_val
    label_file_test = cfg.task1.label_file_test
    
    label_train_val_split = cfg.task1.label_train_val_split

    #output_dir = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/val'
    #label_file = '/mnt/c/Users/salthamm/Documents/phd/data/coliee2021/task1/train/train_labels.json'
    label_file = label_file_train_original

    lengths, dict_paragraphs, failed_files = read_in_para_lengths(corpus_dir, output_dir)
    label_dict = preprocess_label_file(label_file)
    label_keys_list = list(label_dict.keys())
    total_labels = len(label_keys_list)
    labels_train = {}
    labels_val = {}
    for idx, label in enumerate(label_keys_list):
        if idx < (total_labels-label_train_val_split):
            labels_train[label] = label_dict[label]
        else:
            labels_val[label] = label_dict[label]
    
    with open(label_file_train, "w") as fp:
        json.dump(labels_train, fp, indent=4)
    with open(label_file_val, "w") as fp:
        json.dump(labels_val, fp, indent=4)

    analyze_corpus_in_numbers(lengths, dict_paragraphs, label_dict, output_dir)
    intro_often, summ_often, para_often = analyze_text_passages(dict_paragraphs, 50)
    # these text fragments will be removed from the files as they are not considered informative!
    with open(os.path.join(pickle_dir, 'intro_text_often.pkl'), 'wb') as f:
       pickle.dump(intro_often, f)
    with open(os.path.join(pickle_dir, 'summ_text_often.pkl'), 'wb') as f:
       pickle.dump(summ_often, f)
    with open(os.path.join(pickle_dir, 'para_text_often.pkl'), 'wb') as f:
       pickle.dump(para_often, f)

    # in read in docs the non-informative intro from intro_often, and summaries from summary_often are removed
    dict_paragraphs, failed_files = read_in_docs(corpus_dir, output_dir, pickle_dir, removal=True)

    jsonl_index_whole_doc(output_dir, dict_paragraphs)
    jsonl_index_doc_only_para(output_dir, dict_paragraphs)
    jsonl_index_para_separately(output_dir, dict_paragraphs,
                                intro_summ=False)  # without summary and intro
    jsonl_index_para_separately(output_dir, dict_paragraphs,
                                intro_summ=True)  # with summary and intro as separate paragraph

if __name__ == "__main__":
    main()




