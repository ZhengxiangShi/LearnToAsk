import sys
import os
import json
import argparse 
import random
from xmlrpc.client import boolean
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
from builder.dataloader_with_glove import BuilderDataset, RawInputs
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

class MineCraft(Dataset):
    def __init__(self, data_items):
        self.inputs = []
        for i, data in enumerate(tqdm(data_items)):
            encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data 
            """
            encoder_inputs: [max_length]
            grid_repr_inputs: [act_len, 8, 11, 9, 11]
            action_repr_inputs: [act_len, 11]
            location_mask: [act_len, 1089]
            labels: [act_len, 7]
            """
            for action_j in range(len(labels)):
                self.inputs.append((
                    encoder_inputs,  # [100]
                    grid_repr_inputs[action_j], # [8, 11, 9, 11]
                    action_repr_inputs[action_j], # [7]
                    labels[action_j],
                    location_mask[action_j]
                ))
                        
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
    
    traindataset = BuilderDataset(args, split='train', encoder_vocab=None)
    train_items = traindataset.items
    train_dataset = MineCraft(train_items)
    traindataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    print('Finish training data set loading.\n')
    valdataset = BuilderDataset(args, split='val', encoder_vocab=None)
    valid_items = valdataset.items

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

    for epoch in range(train_config['num_epochs']):
        print("Training...")
        model.train()
        train_loss = 0
        total_color = 0
        total_location = 0
        total_color_correct = 0
        total_location_correct = 0
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
            batch_loss, acc, _ = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device)) 

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
            total_actions += len(labels)

        train_loss = train_loss / len(train_items)
        
        print('Train | Loss: {}'.format(train_loss))
        print('Train | Location Acc: {}, Action Type Acc: {}, Color Acc: {}'.format(total_location_correct/total_location, total_action_type_correct/total_actions, total_color_correct/total_color))
        f_output.write('Epoch {}\n'.format(epoch))  
        f_output.write('Train | Loss: {}\n'.format(train_loss))
        f_output.write('Train | Location Acc: {}, Action Type Acc: {}, Color Acc: {}\n'.format(total_location_correct/total_actions, total_action_type_correct/total_actions, total_color_correct/total_color))

        model.eval()
        valid_loss = 0
        max_f1 = 0
        valid_pred_seqs = []
        valid_raw_inputs = []

        with torch.no_grad():
            valid_total_color = 0
            valid_total_location = 0
            valid_total_color_correct = 0
            valid_total_location_correct = 0
            valid_total_action_type_correct = 0
            valid_total_actions = 0
            for i, data in enumerate(tqdm(valid_items)):
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask, raw_input = data
                encoder_inputs, grid_repr_inputs, action_repr_inputs, labels, location_mask = encoder_inputs.unsqueeze(0), grid_repr_inputs.unsqueeze(0), action_repr_inputs.unsqueeze(0), labels.unsqueeze(0), location_mask.unsqueeze(0)
                """
                encoder_inputs: [batch_size, max_length]
                grid_repr_inputs: [batch_size=1, act_len, 8, 11, 9, 11]
                action_repr_inputs: [batch_size=1, act_len, 11]
                location_mask: [batch_size=1, act_len, 1089]
                labels: [batch_size=1, act_len, 7]
                """
                loss, valid_acc, valid_predicted_seq = model(encoder_inputs.long().to(device), grid_repr_inputs.to(device), action_repr_inputs.to(device), labels.long().to(device), location_mask.to(device))
                
                valid_loss += sum(loss)
                valid_total_action_type_correct += valid_acc[0]
                valid_total_location += valid_acc[1]
                valid_total_location_correct += valid_acc[2]
                valid_total_color += valid_acc[3]
                valid_total_color_correct += valid_acc[4]
                valid_total_actions += labels.shape[1]

                valid_pred_seqs.append(valid_predicted_seq)
                valid_raw_inputs.append(raw_input)

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
        
        print('Valid | Recall: {}, Precision: {}, F1: {}, Loss: {}'.format(val_action_recall, val_action_precision, val_action_f1, valid_loss))
        print('Valid | Location Acc: {}, Action Type Acc: {}, Color Acc: {}\n\n\n'.format(valid_total_location_correct/valid_total_location, valid_total_action_type_correct/valid_total_actions, valid_total_color_correct/valid_total_color))
        f_output.write('Valid | Recall: {}, Precision: {}, F1: {}, Loss: {}\n'.format(val_action_recall, val_action_precision, val_action_f1, valid_loss))
        f_output.write('Valid | Location Acc: {}, Action Type Acc: {}, Color Acc: {}\n\n\n'.format(valid_total_location_correct/valid_total_actions, valid_total_action_type_correct/valid_total_actions, valid_total_color_correct/valid_total_color))
        
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
    parser.add_argument('--json_data_dir', type=str, default="./builder_data/data_maxlength100") 
    parser.add_argument('--load_items', default=True, action='store_false')
    parser.add_argument('--from_trained', default='')

    args = parser.parse_args()
    with open(args.config_path, "r") as fp:
        config = json.load(fp)
    config['train_config']['lr'] = args.lr
    config['train_config']['json_data_dir'] = args.json_data_dir
    config['seed'] = args.seed
    seed_torch(args.seed)
    main(args, config)

