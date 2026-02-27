from glob import glob
import os
import nltk
from nltk.stem import WordNetLemmatizer
import json
import spacy
from tqdm import tqdm
import warnings
import argparse
nlp = spacy.load("en_core_web_lg")
warnings.filterwarnings("ignore", category=UserWarning)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--word_association", type=str, default='./data/amber_disc/relation.json')
    parser.add_argument("--safe_words", type=str, default='./data/amber_disc/safe_words.txt')
    parser.add_argument("--annotation", type=str, default='./data/amber_disc/annotations.json')
    parser.add_argument("--metrics", type=str, default='./data/amber_disc/metrics.txt')
    parser.add_argument("--similarity_score", type=float, default=0.8)
    parser.add_argument('--evaluation_type', default='d', choices=['a', 'g', 'd', 'de', 'da', 'dr'])
    args = parser.parse_args()
    return args


def check_synonyms_word(word1, word2, similarity_score):
    token1 = nlp(word1)
    token2 = nlp(word2)
    similarity = token1.similarity(token2)
    return similarity > similarity_score


def extract_nouns(text):
    lemmatizer = WordNetLemmatizer()
    tokens = nltk.word_tokenize(text)
    tagged = nltk.pos_tag(tokens)
    nouns = [lemmatizer.lemmatize(word) for word, pos in tagged if pos.startswith('NN')]
    return nouns


def init(args):
    metrics = {}
    with open(args.metrics, "r") as file:
        lines = file.readlines()

    for line in lines:
        parts = line.strip().split('=')
        if len(parts) == 2:
            variable_name = parts[0].strip()
            variable_value = eval(parts[1].strip())
            metrics[variable_name] = variable_value
            
    return metrics


def compute_amber_disc_score(input_path: str):
    args = get_args()
    args.input_path = input_path
    metrics = init(args)
    association = json.load(open(args.word_association, 'r', encoding='utf-8'))
    hallucination_words = []
    for word1 in association.keys():
        hallucination_words.append(word1)
        for word2 in association[word1]:
            hallucination_words.append(word2)
            
    global_safe_words = []
    with open(args.safe_words, 'r', encoding='utf-8') as safe_file:
        for line in safe_file:
            line = line.split('\n')[0]
            global_safe_words.append(line)

    dimension = {'g': False,'de': False, 'da': False, 'dr': False}
    if args.evaluation_type == 'a':
        for key in dimension.keys():
            dimension[key] = True
    elif args.evaluation_type == 'g':
        dimension['g'] = True
    elif args.evaluation_type == 'd':
        dimension['de'] = True
        dimension['da'] = True
        dimension['dr'] = True
    else:
        dimension[args.evaluation_type] = True
    
    input_path = args.input_path

    with open(input_path, 'r') as f:
        inference_data = json.load(f)
    
    ground_truth = json.load(open(args.annotation, 'r', encoding='utf-8'))

    for i in tqdm(range(len(inference_data))):
        
        id = int(inference_data[i]['data_id'].split('_')[-1])
        
        if ground_truth[id-1]['type'] == 'generative':
            nouns = extract_nouns(inference_data[i]['prediction'])
            after_process_nouns = []
            for noun in nouns:
                if noun in hallucination_words:
                    after_process_nouns.append(noun)
            
            safe_words = []
            safe_list = []
            for idx, word in enumerate(ground_truth[id-1]['truth']):
                safe_words += association[word]
                safe_list += [idx] * len(association[word])
                
            ha_words = []
            ha_list = []
            for idx, word in enumerate(ground_truth[id-1]['hallu']):
                ha_words += association[word]
                ha_list += [idx] * len(association[word])
            
            safe_words += ground_truth[id-1]['truth']
            safe_len = len(ground_truth[id-1]['truth'])
            safe_list += [0] * safe_len
            safe_flag_list = [0] * len(after_process_nouns)
            
            ha_words += ground_truth[id-1]['hallu']
            ha_len = len(ground_truth[id-1]['hallu'])
            ha_list += [0] * ha_len
            
            for idx, noun in enumerate(after_process_nouns):
                if noun in global_safe_words:
                    continue
                
                if noun in safe_words:
                    for j in range(len(safe_words)):
                        if noun == safe_words[j]:
                            if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                            else:
                                safe_list[j] = 1
                            break
                    continue
                
                if noun in ha_words:
                    for j in range(len(ha_words)):
                        if noun == ha_words[j]:
                            if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                            else:
                                ha_list[j] = 1
                            break
                
                for j, check_word in enumerate(ha_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        if j < (len(ha_list) - ha_len):
                                ha_list[ha_list[j] + len(ha_list) - ha_len] = 1
                        else:
                            ha_list[j] = 1
                        break
                
                flag = False
                for j, check_word in enumerate(safe_words):
                    if check_synonyms_word(noun, check_word, args.similarity_score):
                        flag = True
                        if j < (len(safe_list) - safe_len):
                                safe_list[safe_list[j] + len(safe_list) - safe_len] = 1
                        else:
                            safe_list[j] = 1
                        break
                if flag == True:
                    continue
            
                safe_flag_list[idx] = 1

            metrics['chair_score'] += sum(safe_flag_list)
            metrics['chair_num'] += len(safe_flag_list)
            metrics['safe_cover_score'] += sum(safe_list[-safe_len:])
            metrics['safe_cover_num'] += len(safe_list[-safe_len:])
            metrics['hallu_cover_score'] += sum(ha_list[-ha_len:])
            metrics['hallu_cover_num'] += len(ha_list[-ha_len:])
            if sum(safe_flag_list) == 0:
                metrics['non_hallu_score'] += 1
            metrics['non_hallu_num'] += 1
        
        else:
            metrics['qa_correct_num'] += 1
            if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                metrics['as_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                metrics['an_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                metrics['aa_qa_correct_num'] += 1
            elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                metrics['ha_qa_correct_num'] += 1
            else:
                metrics['asso_qa_correct_num'] += 1
            
            truth = ground_truth[id-1]['truth']
            response = inference_data[i]['prediction'].split(',')[0].strip()
            # print(f"ID: {id}, Type: {ground_truth[id-1]['type']}, Truth: {truth}, Response: {response}")
            if truth == 'yes':
                if response == 'Yes':
                    metrics['qa_correct_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_correct_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_correct_score'] += 1
                    else:
                        metrics['asso_qa_correct_score'] += 1
            else:
                metrics['qa_no_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    metrics['as_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    metrics['an_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    metrics['aa_qa_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    metrics['ha_qa_no_num'] += 1
                else:
                    metrics['asso_qa_no_num'] += 1
                
                if response == 'No':
                    metrics['qa_correct_score'] += 1
                    metrics['qa_no_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_correct_score'] += 1
                        metrics['as_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_correct_score'] += 1
                        metrics['an_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_correct_score'] += 1
                        metrics['aa_qa_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_correct_score'] += 1
                        metrics['ha_qa_no_score'] += 1
                    else:
                        metrics['asso_qa_correct_score'] += 1
                        metrics['asso_qa_no_score'] += 1
            
            if response == 'No':
                metrics['qa_ans_no_num'] += 1
                if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                    metrics['as_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                    metrics['an_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                    metrics['aa_qa_ans_no_num'] += 1
                elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                    metrics['ha_qa_ans_no_num'] += 1
                else:
                    metrics['asso_qa_ans_no_num'] += 1
                if truth == 'no':
                    metrics['qa_ans_no_score'] += 1
                    if ground_truth[id-1]['type'] == 'discriminative-attribute-state':
                        metrics['as_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-number':
                        metrics['an_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-attribute-action':
                        metrics['aa_qa_ans_no_score'] += 1
                    elif ground_truth[id-1]['type'] == 'discriminative-hallucination':
                        metrics['ha_qa_ans_no_score'] += 1
                    else:
                        metrics['asso_qa_ans_no_score'] += 1

    results = {}

    if dimension['g']:
        CHAIR = round(metrics['chair_score'] / metrics['chair_num'] * 100, 1)
        Cover = round(metrics['safe_cover_score'] / metrics['safe_cover_num'] * 100, 1)
        Ha = round(metrics['hallu_cover_score'] / metrics['hallu_cover_num'] * 100, 1)
        Ha_p = round(100 - metrics['non_hallu_score'] / metrics['non_hallu_num'] * 100, 1)
   
        results['generative'] = {
            'CHAIR': CHAIR,
            'Cover': Cover,
            'Hal': Ha_p,
            'Cog': Ha,
        }

    if dimension['de'] and dimension['da'] and dimension['dr']:
        Accuracy = round(metrics['qa_correct_score'] / metrics['qa_correct_num'] * 100, 1)
        Precision = round(metrics['qa_ans_no_score'] / metrics['qa_ans_no_num'] * 100, 1)
        Recall = round(metrics['qa_no_score'] / metrics['qa_no_num'] * 100, 1)
        F1 = round(2 * (Precision/100) * (Recall/100) / ((Precision/100) + (Recall/100) + 0.0001) * 100, 1)

        results['discriminative'] = {
            'Accuracy': Accuracy,
            'Precision': Precision,
            'Recall': Recall,
            'F1': F1,
        }
    
    if dimension['de']:
        hallucination_Accuracy = round(metrics['ha_qa_correct_score'] / metrics['ha_qa_correct_num'] * 100, 1)
        hallucination_Precision = round(metrics['ha_qa_ans_no_score'] / metrics['ha_qa_ans_no_num'] * 100, 1)
        hallucination_Recall = round(metrics['ha_qa_no_score'] / metrics['ha_qa_no_num'] * 100, 1)
        hallucination_F1 = round(2 * (hallucination_Precision/100) * (hallucination_Recall/100) / ((hallucination_Precision/100) + (hallucination_Recall/100) + 0.001) * 100, 1)

        results['discriminative-hallucination'] = {
            'Accuracy': hallucination_Accuracy,
            'Precision': hallucination_Precision,
            'Recall': hallucination_Recall,
            'F1': hallucination_F1,
        }
    
    if dimension['da']:
        attr_Accuracy = round((metrics['as_qa_correct_score'] + metrics['an_qa_correct_score'] + metrics['aa_qa_correct_score']) / (metrics['as_qa_correct_num'] + metrics['an_qa_correct_num'] + metrics['aa_qa_correct_num']) * 100, 1)
        attr_Precision = round((metrics['as_qa_ans_no_score'] + metrics['an_qa_ans_no_score'] + metrics['aa_qa_ans_no_score']) / (metrics['as_qa_ans_no_num'] + metrics['an_qa_ans_no_num'] + metrics['aa_qa_ans_no_num']) * 100, 1)
        attr_Recall = round((metrics['as_qa_no_score'] + metrics['an_qa_no_score'] + metrics['aa_qa_no_score']) / (metrics['as_qa_no_num'] + metrics['an_qa_no_num'] + metrics['aa_qa_no_num']) * 100, 1)
        attr_F1 = round(2 * (attr_Precision/100) * (attr_Recall/100) / ((attr_Precision/100) + (attr_Recall/100) + 0.0001) * 100, 1)
        state_Accuracy = round(metrics['as_qa_correct_score'] / metrics['as_qa_correct_num'] * 100, 1)
        state_Precision = round(metrics['as_qa_ans_no_score'] / metrics['as_qa_ans_no_num'] * 100, 1)
        state_Recall = round(metrics['as_qa_no_score'] / metrics['as_qa_no_num'] * 100, 1)
        state_F1 = round(2 * (state_Precision/100) * (state_Recall/100) / ((state_Precision/100) + (state_Recall/100) + 0.0001) * 100, 1)
        number_Accuracy = round(metrics['an_qa_correct_score'] / metrics['an_qa_correct_num'] * 100, 1)
        number_Precision = round(metrics['an_qa_ans_no_score'] / metrics['an_qa_ans_no_num'] * 100, 1)
        number_Recall = round(metrics['an_qa_no_score'] / metrics['an_qa_no_num'] * 100, 1)
        number_F1 = round(2 * (number_Precision/100) * (number_Recall/100) / ((number_Precision/100) + (number_Recall/100) + 0.0001) * 100, 1)
        action_Accuracy = round(metrics['aa_qa_correct_score'] / metrics['aa_qa_correct_num'] * 100, 1)
        action_Precision = round(metrics['aa_qa_ans_no_score'] / metrics['aa_qa_ans_no_num'] * 100, 1)
        action_Recall = round(metrics['aa_qa_no_score'] / metrics['aa_qa_no_num'] * 100, 1)
        action_F1 = round(2 * (action_Precision/100) * (action_Recall/100) / ((action_Precision/100) + (action_Recall/100) + 0.0001) * 100, 1)

        results['discriminative-attribute'] = {
            'Accuracy': attr_Accuracy,
            'Precision': attr_Precision,
            'Recall': attr_Recall,
            'F1': attr_F1,
        }
    
    if dimension['dr']:
        relation_Accuracy = round(metrics['asso_qa_correct_score'] / metrics['asso_qa_correct_num'] * 100, 1)
        relation_Precision = round(metrics['asso_qa_ans_no_score'] / metrics['asso_qa_ans_no_num'] * 100, 1)
        relation_Recall = round(metrics['asso_qa_no_score'] / metrics['asso_qa_no_num'] * 100, 1)
        relation_F1 = round(2 * (relation_Precision/100) * (relation_Recall/100) / ((relation_Precision/100) + (relation_Recall/100) + 0.0001) * 100, 1)
        

        results['discriminative-relation'] = {
            'Accuracy': relation_Accuracy,
            'Precision': relation_Precision,
            'Recall': relation_Recall,
            'F1': relation_F1,
        }

    return results['discriminative']['Accuracy']

