import sys
import os
import json
import argparse 
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from prettytable import PrettyTable
from tensorboardX import SummaryWriter

from builder.vocab import Vocabulary
from builder.model import Builder
from builder.dataloader import BuilderDataset, RawInputs
from builder.utils_builder import evaluate_metrics
from utils import *
from train import MineCraft, computer_acc_each_type

id2action_class = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 2,
}

def main(args, config):
    batch_size = 50
    if args.task_name == 'learn_to_ask':
        testdataset = BuilderDataset(args, split='test', encoder_vocab=None)
        test_items = testdataset.items
        test_dataset = MineCraft(test_items, task_name='learn_to_ask', mode='test')
        testdataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    elif args.task_name == 'learn_to_ask_or_execute':
        testdataset = BuilderDataset(args, split='test', encoder_vocab=None)
        test_items = testdataset.items
    else:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))

    with open(args.encoder_vocab_path, 'rb') as f:
        encoder_vocab = pickle.load(f)

    model = Builder(config, vocabulary=encoder_vocab).to(device)
    model.load_state_dict(torch.load(os.path.join(args.saved_models_path, "model.pt")))

    model.eval()
    test_loss = 0
    if args.task_name == 'learn_to_ask':
        with torch.no_grad():
            test_total_actions = 0
            test_correct = 0
            predicted_seq = []
            ground_truth_seq = []

            for i, data in enumerate(tqdm(testdataloader)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = data
                """
                encoder_inputs: [batch_size, max_length]
                grid_repr_inputs: [batch_size, 8, 11, 9, 11]
                action_repr_inputs: [batch_size, 11]
                labels: [batch_size]
                location_mask: [batch_size, 1089]
                """
                loss, action_type_logits = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device))
                
                test_loss += loss
                test_correct_batch = (torch.argmax(action_type_logits.cpu(), dim=-1) == labels).sum()
                test_correct += test_correct_batch.item()
                test_total_actions += len(labels)
                predicted_seq.extend(torch.argmax(action_type_logits.cpu(), dim=-1).tolist())
                ground_truth_seq.extend(labels.tolist())
            
            each_type_acc = computer_acc_each_type(predicted_seq, ground_truth_seq)
            test_loss = test_loss / test_total_actions
            test_acc = test_correct / test_total_actions
        
        print('Test | Loss: {}, Acc: {}'.format(test_loss, test_acc))
        print('Acc', each_type_acc)
        print('Type Acc:\n', round(each_type_acc[0][0]/sum(each_type_acc[0]),4), round(each_type_acc[0][1]/sum(each_type_acc[0]),4), round(each_type_acc[0][2]/sum(each_type_acc[0]),4),'\n',\
            round(each_type_acc[1][0]/sum(each_type_acc[1]),4), round(each_type_acc[1][1]/sum(each_type_acc[1]),4), round(each_type_acc[1][2]/sum(each_type_acc[1]),4),'\n',\
            round(each_type_acc[2][0]/sum(each_type_acc[2]),4), round(each_type_acc[2][1]/sum(each_type_acc[2]),4), round(each_type_acc[2][2]/sum(each_type_acc[2]),4))
    
    elif args.task_name == 'learn_to_ask_or_execute':
        test_pred_seqs = []
        test_raw_inputs = []
        with torch.no_grad():
            test_total_color = 0
            test_total_location = 0
            test_total_clarification = 0
            test_total_color_correct = 0
            test_total_location_correct = 0
            test_total_clarification_correct = 0
            test_total_action_type_correct = 0
            test_total_actions = 0
            predicted_action_type_seq = []
            oracle_action_type_seq = []
            for i, data in enumerate(tqdm(test_items)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = encoder_inputs.unsqueeze(0), grid_repr_inputs.unsqueeze(0), action_repr_inputs.unsqueeze(0), labels.unsqueeze(0), location_mask.unsqueeze(0)
                """
                encoder_inputs: [batch_size=1, max_length]
                grid_repr_inputs: [batch_size=1, act_len, 8, 11, 9, 11]
                action_repr_inputs: [batch_size=1, act_len, 11]
                location_mask: [batch_size=1, act_len, 1089]
                labels: [batch_size=1, act_len, 7]
                """
                loss, test_acc, test_predicted_seq = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), args.task_name)
                
                test_loss += sum(loss)
                test_total_action_type_correct += test_acc[0]
                test_total_location += test_acc[1]
                test_total_location_correct += test_acc[2]
                test_total_color += test_acc[3]
                test_total_color_correct += test_acc[4]
                test_total_clarification += test_acc[5]
                test_total_clarification_correct += test_acc[6]
                test_total_actions += labels.shape[1]
                
                if (labels[0,0,1] == 0 or labels[0,0,1] == 1 or labels[0,0,1] == 2):
                    test_pred_seqs.append(test_predicted_seq)
                    test_raw_inputs.append(raw_input)
            
                predicted_action_type_seq.append(id2action_class[test_predicted_seq[0][1].item()])
                oracle_action_type_seq.append(id2action_class[labels[0, 0, 1].item()])

            each_type_acc = computer_acc_each_type(predicted_action_type_seq, oracle_action_type_seq)
            val_action_precision, val_action_recall, val_action_f1 = evaluate_metrics(test_pred_seqs, test_raw_inputs)
            test_loss = test_loss / len(test_items) 

        
        print('test | Recall: {}, Precision: {}, F1: {}, Loss: {}'.format(val_action_recall, val_action_precision, val_action_f1, test_loss))
        print('test | Location Acc: {}, Action Type Acc: {}, Color Acc: {}, Clar Acc: {}'.format(test_total_location_correct/test_total_location, test_total_action_type_correct/test_total_actions, test_total_color_correct/test_total_color, test_total_clarification_correct/test_total_clarification))    
        print('Type Acc:\n', round(each_type_acc[0][0]/sum(each_type_acc[0]),4), round(each_type_acc[0][1]/sum(each_type_acc[0]),4), round(each_type_acc[0][2]/sum(each_type_acc[0]),4),'\n',\
            round(each_type_acc[1][0]/sum(each_type_acc[1]),4), round(each_type_acc[1][1]/sum(each_type_acc[1]),4), round(each_type_acc[1][2]/sum(each_type_acc[1]),4),'\n',\
            round(each_type_acc[2][0]/sum(each_type_acc[2]),4), round(each_type_acc[2][1]/sum(each_type_acc[2]),4), round(each_type_acc[2][2]/sum(each_type_acc[2]),4))


    else:
        pass

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--saved_models_path', type=str, default='./default_path', help='path for saving trained models')
    parser.add_argument('--encoder_vocab_path', type=str, default='./builder_data/vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl')
    # Args for dataset
    parser.add_argument('--json_data_dir', type=str, default="./builder_data/builder_with_questions_data") 
    parser.add_argument('--task_name', type=str, default="learn_to_ask") 
    parser.add_argument('--load_items', default=True)

    args = parser.parse_args()
    with open(os.path.join(args.saved_models_path, "config.json"), "r") as fp:
        config = json.load(fp)
    main(args, config)