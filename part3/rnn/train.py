from transformers import AutoModelForCausalLM, AutoConfig
from transformers.models.llama.modeling_llama import LlamaAttention, LlamaForCausalLM, LlamaMLP 
from transformers.models.llama.configuration_llama import LlamaConfig

import torch
import torch.nn as nn
import numpy as np
from transformers import get_scheduler
from torch.optim import AdamW
from torch.utils.data import DataLoader

from tqdm import tqdm
# import wandb
import os
import json

from utils import set_seed, parse_args
from data import load_dataset, IsTreeDataset
from model import get_transformer_model, get_rnn_model, get_hybrid_model

open('train.log', 'w').close()
def print(*args, **kwargs):
    with open('train.log', 'a') as f:
        __builtins__.print(*args, **kwargs, file=f)

import time
class Timer:
    def __init__(self,desc):
        self.desc = desc
        
    def __enter__(self):
        self.start = time.time()
        return self
    
    def __exit__(self, *args):
        print(f'{self.desc} took {time.time() - self.start} seconds')

@torch.no_grad()
def evaluate(model, val_loader, args):
    if not isinstance(model, nn.DataParallel):
        model = nn.DataParallel(model)
    model.eval()
    total_loss = 0
    total_correct_samples = 0
    total_samples = 0
    with torch.no_grad():
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        for batch in val_loader:
        # for batch in tqdm(val_loader, total=len(val_loader)):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            # input_ids = input_ids.to(args.device)
            # attention_mask = attention_mask.to(args.device)
            # labels = labels.to(args.device)
            if args.model_type == 'transformer':
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                logits = outputs.logits
            else:
                logits = model(input_ids)
                labels = labels.to(args.device)
                loss = criterion(logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1))
                
            logits = logits[..., :-1, :].argmax(dim=-1)
            total_loss += loss.item()

            mask = labels[..., 1:] != -100
            
            # Compute accuracy per sample
            sample_acc = (logits == labels[..., 1:]) | ~mask  # True for correct predictions and ignored tokens
            sample_acc = sample_acc.all(dim=-1)  # Check if all tokens in a sample are correct/ignored
            total_correct_samples += sample_acc.sum().item()
            total_samples += labels.size(0)  # Count each sample in the batch
    model.train()
    return total_loss / len(val_loader), total_correct_samples / total_samples

def save_to_status_file(data, output_dir):
    with open(f'{output_dir}/status.json', 'w') as f:
        json.dump(data, f)
        
def get_from_status_file(output_dir):
    if os.path.exists(f'{output_dir}/status.json'):
        with open(f'{output_dir}/status.json', 'r') as f:
            return json.load(f)
    raise NotImplementedError

def remove_additional_checkpt(output_dir):
    for file in os.listdir(output_dir):
        if file.endswith('.pt') and 'best' not in file:
            os.remove(os.path.join(output_dir, file))

def load_logs_from_json(output_dir):
    if os.path.exists(os.path.join(output_dir, 'results.json')):
        return json.load(open(os.path.join(output_dir, 'results.json')))
    print('WARNING: can\'t find results.json file at ', output_dir)
    return {'train_losses': [], 'train_accs': [], 'val_accs': []}

def train(model, optimizer, scheduler, train_loader, val_loader, args, starting_stats=None):
    model = nn.DataParallel(model)
    model.train()
    print('start training')
    if starting_stats is None:
        total_samples_processed = 0
        step = 0
        train_loss_accumulator = 0
        train_acc_accumulator = 0
        train_samples_accumulator = 0
    else:
        total_samples_processed = starting_stats['total_samples_processed']
        step = starting_stats['step']
        train_loss_accumulator = starting_stats['train_loss_accumulator']
        train_acc_accumulator = starting_stats['train_acc_accumulator']
        train_samples_accumulator = starting_stats['train_samples_accumulator']    
    
    past_results = load_logs_from_json(args.previous_model_path)
    train_losses = past_results['train_losses']
    train_accs = past_results['train_accs']
    val_accs = past_results['val_accs']
    best_val_acc = max(val_accs) if val_accs else 0
    # pbar = tqdm(total=(args.total_training_samples // args.batch_size))
    criterion = nn.CrossEntropyLoss(ignore_index=-100)
    while total_samples_processed < args.total_training_samples:
        for _,batch in zip(range(len(train_loader)),train_loader):
            input_ids, attention_mask, labels = batch['input_ids'], batch['attention_mask'], batch['labels']
            batch_size = input_ids.size(0)
            if total_samples_processed > args.total_training_samples:
                break

            with Timer(f'Batch {step}'):
                # Forward and backward passes
                model.zero_grad()
                optimizer.zero_grad()
                # input_ids = input_ids.to(args.device)
                # attention_mask = attention_mask.to(args.device)
                # labels = labels.to(args.device)
                if args.model_type == 'transformer':
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    logits = outputs.logits
                    loss = outputs.loss
                else:
                    logits = model(input_ids)
                    labels = labels.to(args.device)
                    loss = criterion(logits[..., :-1, :].reshape(-1, logits.size(-1)), labels[..., 1:].reshape(-1))
                loss.backward()
                optimizer.step()
                scheduler.step()
        
            train_loss_accumulator += loss.item()
            logits = logits[..., :-1, :].detach().argmax(dim=-1)
            mask = labels[..., 1:] != -100
            correct_predictions = (logits == labels[..., 1:]) | ~mask
            correct_predictions = correct_predictions.all(dim=-1).sum().item()
            total_predictions = labels.size(0)
            train_acc_accumulator += correct_predictions
            train_samples_accumulator += total_predictions


            # Log and evaluate at log_interval
            if total_samples_processed % args.log_interval < batch_size or total_samples_processed + batch_size >= args.total_training_samples:
                # from IPython import embed; embed()
                train_acc = train_acc_accumulator / train_samples_accumulator if train_samples_accumulator > 0 else 0
                train_loss = train_loss_accumulator / (step + 1)
                val_loss, val_acc = evaluate(model, val_loader, args)
                lr = scheduler.optimizer.param_groups[0]['lr']
                print(f'Step {step} | Samples {total_samples_processed} | Train acc: {train_acc} | Val loss: {val_loss} | Val acc: {val_acc} | learning rate: {lr}')
                if val_acc > best_val_acc:
                    torch.save(model.state_dict(), f'{args.output_dir}/model_best.pt')
                    best_val_acc = val_acc
                # save checkpoint
                save_to_status_file({
                    'total_samples_processed': total_samples_processed,
                    'step': step,
                    'train_loss_accumulator': train_loss_accumulator,
                    'train_acc_accumulator': train_acc_accumulator,
                    'train_samples_accumulator': train_samples_accumulator,
                }, args.output_dir)
                remove_additional_checkpt(args.output_dir)
                torch.save(model.state_dict(), f'{args.output_dir}/model_{step}.pt')
                print('Checkpoint saved successfully at step:', step,'!')
                
                if args.report_to_wandb:
                    wandb.log({"Step": step, "Train Loss": train_loss, "Train Accuracy": train_acc, "Validation Loss": val_loss, "Validation Accuracy": val_acc}, step=step)
                train_losses.append(train_loss)
                train_accs.append(train_acc)
                val_accs.append(val_acc)
                json.dump({'train_losses': train_losses,'train_accs': train_accs, 'val_accs': val_accs}, open(f'{args.output_dir}/results.json', 'w'))
                # Log metrics to wandb
                # Reset training accumulators
                train_loss_accumulator = 0
                train_acc_accumulator = 0
                train_samples_accumulator = 0

            total_samples_processed += batch_size
            step += 1
            # pbar.update(1) 
            if total_samples_processed >= args.total_training_samples:
                break
    # pbar.close()
    if args.model_path:
        torch.save(model.state_dict(), args.model_path)
    json.dump({
        'train_losses': train_losses,
        'train_accs': train_accs,
        'val_accs': val_accs
    }, open(f'{args.output_dir}/results.json', 'w'))

def main():
    args = parse_args()
    print(args.previous_model_path)
    args.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if args.report_to_wandb:
        wandb.init()
    set_seed(args.seed)
    print('Loading train set...')
    train_dataset = load_dataset(args.dataset_dir)
    print('Loading val set...')
    val_dataset = load_dataset(os.path.join(args.dataset_dir, 'val'))

    if args.model_type == 'transformer':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_transformer_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                intermediate_size=config['intermediate_size'],
                num_hidden_layers=config['num_hidden_layers'],
                num_attention_heads=config['num_attention_heads'],
                max_position_embeddings=config['max_position_embeddings']
            )
        else:
            model = get_transformer_model(
                train_dataset,
                hidden_size=128,
                intermediate_size=512,
                num_hidden_layers=20,
                num_attention_heads=8,
                max_position_embeddings=4096
            )
    elif args.model_type == 'rnn':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_rnn_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers']
            )
        else:
            model = get_rnn_model(
                train_dataset,
                hidden_size = 128,
                num_hidden_layers= 10
            )
    elif args.model_type == 'hybrid':
        if args.model_config_path:
            config = json.load(open(args.model_config_path))
            for k, v in config.items():
                config[k] = int(v)
            model = get_hybrid_model(
                train_dataset,
                hidden_size=config['hidden_size'],
                num_hidden_layers=config['num_hidden_layers'],
                max_position_embeddings=config['max_position_embeddings'],
                num_attention_heads=config['num_attention_heads'],
                intermediate_size=config['intermediate_size']
            )
        else:
            model = get_hybrid_model(
                train_dataset,
                hidden_size=128,
                num_hidden_layers=9
            )
    else:
        raise NotImplementedError
    
    starting_stats = None
    if args.previous_model_path is not None and os.path.exists(args.previous_model_path):
        print("Loading model from previous checkpoint", args.previous_model_path)
        files = os.listdir(args.previous_model_path)
        model_path = [file for file in files if file.endswith('.pt')][0]
        print('Find a model file called:', model_path)
        state_dict_with_module = torch.load(os.path.join(args.previous_model_path, model_path))
        model.load_state_dict({k.replace('module.', ''): v for k, v in state_dict_with_module.items()})
        # load status
        starting_stats = get_from_status_file(args.previous_model_path)
        n_batch = starting_stats['step']
        print('Last train time proceed to batch number:', n_batch)
        train_dataset.set_start_idx(n_batch)
        
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
        
    model = model.to(device=args.device, dtype=torch.float32)
    print('model is:',model)
    val_loss, val_acc = evaluate(model, val_loader, args)
    print(f'Initial | val loss: {val_loss} | val acc: {val_acc}')
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = get_scheduler(
        'cosine',
        optimizer=optimizer,
        num_warmup_steps=(args.warmup_samples // args.batch_size),
        num_training_steps=(args.total_training_samples // args.batch_size)
    )
    if args.output_dir:
        os.makedirs(args.output_dir, exist_ok=True)
    train(model, optimizer, scheduler, train_loader, val_loader, args, starting_stats=starting_stats)

if __name__ == '__main__':
    main()