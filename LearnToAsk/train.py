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


def seed_torch(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed) # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True

def count_parameters(model):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params

def computer_acc_each_type(predicted_seq, ground_truth_seq):
    """
    predicted_seq: list
    ground_truth_seq: list
    """
    predicted_seq = np.array(predicted_seq)
    ground_truth_seq = np.array(ground_truth_seq)
    
    execution_seq = predicted_seq[ground_truth_seq == 0]
    clarification_seq = predicted_seq[ground_truth_seq == 1]
    other_seq = predicted_seq[ground_truth_seq == 2]
    return (sum(execution_seq==0), sum(execution_seq==1), sum(execution_seq==2)), \
        (sum(clarification_seq==0), sum(clarification_seq==1), sum(clarification_seq==2)), \
        (sum(other_seq==0), sum(other_seq==1), sum(other_seq==2))

action2id = {
	"placement": 0,
	"removal": 1,
    "stop": 2,
    "clarification": 3,
    "other": 4
}

id2action_class = {
    0: 0, 1: 0, 2: 0, 3: 1, 4: 2,
}


class MineCraft(Dataset):
    def __init__(self, data_items, task_name='learn_to_ask', mode=None):
        self.inputs = []
        if task_name == 'learn_to_ask':
            count_execute = 0
            count_clarification = 0
            count_other = 0
            clarification_tmp = []
            other_tmp = []
            for i, data in enumerate(tqdm(data_items)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data 
                """
                encoder_inputs: [max_length]
                grid_repr_inputs: [act_len, 8, 11, 9, 11]
                action_repr_inputs: [act_len, 11]
                location_mask: [act_len, 1089]
                labels: [act_len, 7]
                """
                action_type_label = {
                    "execute": 0,
                    "clarification": 1,
                    "other": 2
                }
                if labels[0, 1] == 0 or labels[0, 1] == 1 or labels[0, 1] == 2: 
                    ground_truth_action_type = 'execute'
                    count_execute += 1
                elif labels[0, 1] == 3:
                    ground_truth_action_type = 'clarification'
                    count_clarification += 1
                    clarification_tmp.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[0], # [8, 11, 9, 11]
                    action_repr_inputs[0], # [7]
                    action_type_label[ground_truth_action_type],
                    location_mask[0]
                    ))
                elif labels[0, 1] == 4:
                    ground_truth_action_type = 'other'
                    count_other += 1
                    other_tmp.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[0], # [8, 11, 9, 11]
                    action_repr_inputs[0], # [7]
                    action_type_label[ground_truth_action_type],
                    location_mask[0]
                    ))
                else:
                    print('Error')

                self.inputs.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[0], # [8, 11, 9, 11]
                    action_repr_inputs[0], # [7]
                    action_type_label[ground_truth_action_type],
                    location_mask[0]
                ))
            
            if mode == 'train':
                self.inputs.extend(clarification_tmp*(count_execute//count_clarification))
                self.inputs.extend(other_tmp*(count_execute//count_other))
                print('After augmentation:', len(self.inputs))

        elif task_name == 'learn_to_ask_or_execute':
            count_execute = 0
            count_clarification = 0
            count_other = 0
            clarification_tmp = []
            other_tmp = []
            for i, data in enumerate(tqdm(data_items)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data 
                if labels[0, 1] == 0 or labels[0, 1] == 1 or labels[0, 1] == 2: 
                    # print('execute', action_repr_inputs.shape, grid_repr_inputs.shape)
                    ground_truth_action_type = 'execute'
                    count_execute += 1
                elif labels[0, 1] == 3:
                    # print('clarification', action_repr_inputs.shape, grid_repr_inputs.shape)
                    ground_truth_action_type = 'clarification'
                    count_clarification += 1
                    clarification_tmp.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[0], # [8, 11, 9, 11]
                    action_repr_inputs[0], # [7]
                    labels[0],
                    location_mask[0]
                    ))
                elif labels[0, 1] == 4:
                    ground_truth_action_type = 'other'
                    count_other += 1
                    other_tmp.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[0], # [8, 11, 9, 11]
                    action_repr_inputs[0], # [7]
                    labels[0],
                    location_mask[0]
                    ))
                else:
                    print('Error')

                for action_j in range(len(labels)):
                    self.inputs.append((
                        encoder_inputs,  # [100]
                        grid_repr_inputs[action_j], # [8, 11, 9, 11]
                        action_repr_inputs[action_j], # [7]
                        labels[action_j],
                        location_mask[action_j]
                    ))

            if mode == 'train':
                self.inputs.extend(clarification_tmp*(count_execute//count_clarification))
                self.inputs.extend(other_tmp*(count_execute//count_other))
                print('After augmentation:', len(self.inputs))

        else:
            print('Please select a correct task type.')
                        
    def __len__(self):
        return len(self.inputs)
    
    def __getitem__(self, idx):
        return self.inputs[idx]

        
def main(args, config):
    saved_model = args.saved_models_path
    if not os.path.exists(saved_model):
        os.mkdir(saved_model)
    writer = SummaryWriter(log_dir=saved_model)
    f_output = open(os.path.join(saved_model, 'output.txt'), 'w')

    train_config = config["train_config"]
    batch_size = config["train_config"]["batch_size"]
    print("\nHyperparameter configuration written to {}.\n".format(os.path.join(saved_model, 'config.json')))

    with open(args.encoder_vocab_path, 'rb') as f:
        encoder_vocab = pickle.load(f)
    
    if args.task_name == 'learn_to_ask':
        traindataset = BuilderDataset(args, split='train', encoder_vocab=None)
        train_items = traindataset.items
        train_dataset = MineCraft(train_items, task_name='learn_to_ask', mode='train')
        traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        valdataset = BuilderDataset(args, split='val', encoder_vocab=None)
        valid_items = valdataset.items
        valid_dataset = MineCraft(valid_items, task_name='learn_to_ask')
        validdataloader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=True)
        print('Finish data set loading.\n')
    elif args.task_name == 'learn_to_ask_or_execute':
        traindataset = BuilderDataset(args, split='train', encoder_vocab=None)
        train_items = traindataset.items
        train_dataset = MineCraft(train_items, task_name='learn_to_ask_or_execute', mode='train')
        traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)       
        valdataset = BuilderDataset(args, split='val', encoder_vocab=None)
        valid_items = valdataset.items
        print('Finish data set loading.\n')
    else:
        pass

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using {}'.format(device))

    model = Builder(config, vocabulary=encoder_vocab).to(device)

    if args.from_trained:
        model.load_state_dict(torch.load(os.path.join(saved_model, "model.pt")))
    
    print('\n\n')
    total_params_encoder = count_parameters(model.encoder)
    total_params_decoder = count_parameters(model.decoder)
    print('\n\n')
    
    config['total_params_encoder'] = total_params_encoder
    config['total_params_decoder'] = total_params_decoder
    with open(os.path.join(saved_model, 'config.json'), 'w') as f:
        json.dump(config, f)

    optimizer = optim.Adam(model.parameters(), lr=train_config["lr"], betas=(train_config["beta1"], train_config["beta2"]))

    if args.task_name == 'learn_to_ask_or_execute':
        for epoch in range(train_config['num_epochs']):
            print("Training...")
            model.train()
            train_loss = 0
            total_color = 0
            total_location = 0
            total_clarification = 0
            total_color_correct = 0
            total_location_correct = 0
            total_clarification_correct = 0
            total_action_type_correct = 0
            total_actions = 0

            for i, data in enumerate(tqdm(traindataloader)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = data 
                """
                encoder_inputs: [batch_size, max_length]
                grid_repr_inputs: [batch_size, 8, 11, 9, 11]
                action_repr_inputs: [batch_size, 11]
                labels: [batch_size, 7]
                location_mask: [batch_size, 1089]
                """
                batch_loss, acc, _ = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), args.task_name) 

                batch_loss = sum(batch_loss)
                optimizer.zero_grad()
                batch_loss.backward()
                optimizer.step()
                train_loss += batch_loss

                total_action_type_correct += acc[0]
                total_location += acc[1]
                total_location_correct += acc[2]
                total_color += acc[3]
                total_color_correct += acc[4]
                total_clarification += acc[5]
                total_clarification_correct += acc[6]
                total_actions += len(labels)

            train_loss = train_loss / len(train_items)
            
            print('Epoch {}'.format(epoch))
            print('Train | Loss: {}'.format(train_loss))
            print('Train | Location Acc: {}, Action Type Acc: {}, Color Acc: {}, Clar Acc: {}'.format(total_location_correct/total_location, total_action_type_correct/total_actions, total_color_correct/total_color, total_clarification_correct/total_clarification))
            f_output.write('Epoch {}\n'.format(epoch))  
            f_output.write('Train | Loss: {}\n'.format(train_loss)) 
            f_output.write('Train | Location Acc: {}, Action Type Acc: {}, Color Acc: {}, Clar Acc: {}\n'.format(total_location_correct/total_location, total_action_type_correct/total_actions, total_color_correct/total_color, total_clarification_correct/total_clarification))
            
            model.eval()
            valid_loss = 0
            max_f1 = 0
            valid_pred_seqs = []
            valid_raw_inputs = []
            with torch.no_grad():
                valid_total_color = 0
                valid_total_location = 0
                valid_total_clarification = 0
                valid_total_color_correct = 0
                valid_total_location_correct = 0
                valid_total_clarification_correct = 0
                valid_total_action_type_correct = 0
                valid_total_actions = 0
                predicted_action_type_seq = []
                oracle_action_type_seq = []
                for i, data in enumerate(tqdm(valid_items)):
                    encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data
                    encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = encoder_inputs.unsqueeze(0), grid_repr_inputs.unsqueeze(0), action_repr_inputs.unsqueeze(0), labels.unsqueeze(0), location_mask.unsqueeze(0)
                    """
                    encoder_inputs: [batch_size=1, max_length]
                    grid_repr_inputs: [batch_size=1, act_len, 8, 11, 9, 11]
                    action_repr_inputs: [batch_size=1, act_len, 11]
                    location_mask: [batch_size=1, act_len, 1089]
                    labels: [batch_size=1, act_len, 7]
                    """
                    loss, valid_acc, valid_predicted_seq = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), args.task_name)
                    
                    valid_loss += sum(loss)
                    valid_total_action_type_correct += valid_acc[0]
                    valid_total_location += valid_acc[1]
                    valid_total_location_correct += valid_acc[2]
                    valid_total_color += valid_acc[3]
                    valid_total_color_correct += valid_acc[4]
                    valid_total_clarification += valid_acc[5]
                    valid_total_clarification_correct += valid_acc[6]
                    valid_total_actions += labels.shape[1]
                    
                    if (labels[0,0,1] == 0 or labels[0,0,1] == 1 or labels[0,0,1] == 2):
                        valid_pred_seqs.append(valid_predicted_seq)
                        valid_raw_inputs.append(raw_input)
                
                    predicted_action_type_seq.append(id2action_class[valid_predicted_seq[0][1].item()])
                    oracle_action_type_seq.append(id2action_class[labels[0, 0, 1].item()])

                each_type_acc = computer_acc_each_type(predicted_action_type_seq, oracle_action_type_seq)
                val_action_precision, val_action_recall, val_action_f1 = evaluate_metrics(valid_pred_seqs, valid_raw_inputs)
                valid_loss = valid_loss / len(valid_items) 

                if val_action_f1 > max_f1:
                    print("Saving model at {}".format(os.path.join(saved_model, "model.pt")))
                    torch.save(model.state_dict(), os.path.join(saved_model, "model.pt"))
                    max_f1 = val_action_f1
            
            writer.add_scalars("Recall", { "validation": val_action_recall}, epoch)
            writer.add_scalars("Precision", {"validation": val_action_precision}, epoch)
            writer.add_scalars("F1", {"validation": val_action_f1}, epoch)
            writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)
            writer.add_scalars("Clarification", {"train": total_clarification_correct/total_clarification, "validation": valid_total_clarification_correct/valid_total_clarification}, epoch)
            
            print('Valid | Recall: {}, Precision: {}, F1: {}, Loss: {}'.format(val_action_recall, val_action_precision, val_action_f1, valid_loss))
            print('Valid | Location Acc: {}, Action Type Acc: {}, Color Acc: {}, Clar Acc: {}'.format(valid_total_location_correct/valid_total_location, valid_total_action_type_correct/valid_total_actions, valid_total_color_correct/valid_total_color, valid_total_clarification_correct/valid_total_clarification))    
            print('Type Acc:\n', round(each_type_acc[0][0]/sum(each_type_acc[0]),4), round(each_type_acc[0][1]/sum(each_type_acc[0]),4), round(each_type_acc[0][2]/sum(each_type_acc[0]),4),'\n',\
                round(each_type_acc[1][0]/sum(each_type_acc[1]),4), round(each_type_acc[1][1]/sum(each_type_acc[1]),4), round(each_type_acc[1][2]/sum(each_type_acc[1]),4),'\n',\
                round(each_type_acc[2][0]/sum(each_type_acc[2]),4), round(each_type_acc[2][1]/sum(each_type_acc[2]),4), round(each_type_acc[2][2]/sum(each_type_acc[2]),4))
            f_output.write('Valid | Recall: {}, Precision: {}, F1: {}, Loss: {}\n'.format(val_action_recall, val_action_precision, val_action_f1, valid_loss))
            f_output.write('Valid | Location Acc: {}, Action Type Acc: {}, Color Acc: {}, Clar Acc: {}\n'.format(valid_total_location_correct/valid_total_location, valid_total_action_type_correct/valid_total_actions, valid_total_color_correct/valid_total_color, valid_total_clarification_correct/valid_total_clarification))    
            f_output.write('Type Acc:\n{} {} {},\n{} {} {},\n{} {} {}\n\n\n'.format(round(each_type_acc[0][0]/sum(each_type_acc[0]),4), round(each_type_acc[0][1]/sum(each_type_acc[0]),4), round(each_type_acc[0][2]/sum(each_type_acc[0]),4),\
                round(each_type_acc[1][0]/sum(each_type_acc[1]),4), round(each_type_acc[1][1]/sum(each_type_acc[1]),4), round(each_type_acc[1][2]/sum(each_type_acc[1]),4),\
                round(each_type_acc[2][0]/sum(each_type_acc[2]),4), round(each_type_acc[2][1]/sum(each_type_acc[2]),4), round(each_type_acc[2][2]/sum(each_type_acc[2]),4)))

    elif args.task_name == 'learn_to_ask':
        for epoch in range(train_config['num_epochs']):
            print("Training...")
            model.train()
            train_loss = 0
            total_clarification = 0
            total_clarification_correct = 0
            total_action_type_correct = 0
            total_actions = 0
            correct = 0

            for i, data in enumerate(tqdm(traindataloader)): 
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = data 
                """
                encoder_inputs: [batch_size, max_length]
                grid_repr_inputs: [batch_size, 8, 11, 9, 11]
                action_repr_inputs: [batch_size, 11]
                labels: [batch_size]
                location_mask: [batch_size, 1089]
                """
                loss, action_type_logits = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), args.task_name) 
               
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                train_loss += loss
                correct_batch = (torch.argmax(action_type_logits.cpu(), dim=-1) == labels).sum()
                correct += correct_batch.item()
                total_actions += len(labels)
            
            train_loss = train_loss / total_actions
            train_acc = correct / total_actions
            
            print('Train | Loss: {}, Acc: {}'.format(train_loss, train_acc))

            model.eval()
            valid_loss = 0
            max_acc = 0
            with torch.no_grad():
                valid_total_clarification = 0
                valid_total_clarification_correct = 0
                valid_total_action_type_correct = 0
                valid_total_actions = 0
                valid_correct = 0
                predicted_seq = []
                ground_truth_seq = []

                for i, data in enumerate(tqdm(validdataloader)):
                    encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = data
                    """
                    encoder_inputs: [batch_size, max_length]
                    grid_repr_inputs: [batch_size, 8, 11, 9, 11]
                    action_repr_inputs: [batch_size, 11]
                    labels: [batch_size]
                    location_mask: [batch_size, 1089]
                    """
                    loss, action_type_logits = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device), args.task_name)
                    
                    valid_loss += loss
                    valid_correct_batch = (torch.argmax(action_type_logits.cpu(), dim=-1) == labels).sum()
                    valid_correct += valid_correct_batch.item()
                    valid_total_actions += len(labels)
                    predicted_seq.extend(torch.argmax(action_type_logits.cpu(), dim=-1).tolist())
                    ground_truth_seq.extend(labels.tolist())
                
                each_type_acc = computer_acc_each_type(predicted_seq, ground_truth_seq)
                valid_loss = valid_loss / valid_total_actions
                valid_acc = valid_correct / valid_total_actions

                if valid_acc > max_acc:
                    print("Saving model at {}".format(os.path.join(saved_model, "model.pt")))
                    torch.save(model.state_dict(), os.path.join(saved_model, "model.pt"))
                    max_acc = valid_acc
            
            writer.add_scalars("Loss", {"train": train_loss, "validation": valid_loss}, epoch)
            writer.add_scalars("Acc", {"train": train_acc, "validation": valid_acc}, epoch)
            
            print('Valid | Loss: {}, Acc: {}'.format(valid_loss, valid_acc))
            print('Acc', each_type_acc)
            print('Type Acc:\n', round(each_type_acc[0][0]/sum(each_type_acc[0]),4), round(each_type_acc[0][1]/sum(each_type_acc[0]),4), round(each_type_acc[0][2]/sum(each_type_acc[0]),4),'\n',\
                round(each_type_acc[1][0]/sum(each_type_acc[1]),4), round(each_type_acc[1][1]/sum(each_type_acc[1]),4), round(each_type_acc[1][2]/sum(each_type_acc[1]),4),'\n',\
                round(each_type_acc[2][0]/sum(each_type_acc[2]),4), round(each_type_acc[2][1]/sum(each_type_acc[2]),4), round(each_type_acc[2][2]/sum(each_type_acc[2]),4))
    else:
        print('Please select a correct task type.')

    writer.close()
    f_output.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--config_path', type=str, default='./builder/config.json')
    parser.add_argument('--saved_models_path', type=str, default='./default_path', help='path for saving trained models')
    parser.add_argument('--encoder_vocab_path', type=str, default='./builder_data/vocabulary/glove.42B.300d-lower-1r-speaker-oov_as_unk-all_splits/vocab.pkl')
    # Args for dataset
    parser.add_argument('--json_data_dir', type=str, default="./builder_data/builder_with_questions_data") 
    parser.add_argument('--task_name', type=str, default="learn_to_ask") # learn_to_ask_or_execute
    parser.add_argument('--load_items', default=True)
    parser.add_argument('--from_trained', default='')
    parser.add_argument('--beta_0', type=float, default=0.1)
    parser.add_argument('--beta_1', type=float, default=0.8)
    parser.add_argument('--beta_2', type=float, default=0.1)

    args = parser.parse_args()
    with open(args.config_path, "r") as fp:
        config = json.load(fp)
    if args.task_name == 'learn_to_ask_or_execute':
        config['decoder_config']['action_type_size'] = 5
    config['decoder_config']['loss_weight'] = (args.beta_0, args.beta_1, args.beta_2)
    config['train_config']['lr'] = args.lr
    config['train_config']['json_data_dir'] = args.json_data_dir
    config['train_config']['task_name'] = args.task_name
    config['seed'] = args.seed
    seed_torch(args.seed)
    main(args, config)

