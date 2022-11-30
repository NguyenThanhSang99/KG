import torch
import argparse
import numpy as np
import random
from dataloader import DataLoader
import torch.optim as optim
from tqdm import tqdm
from time import time
from model import TransH
import sys

from utils.log_utils import *


def train(args):
    fix_seed(args.seed)

    log_save_id = create_log_id(args.save_dir)
    logging_config(folder=args.save_dir, name='log{:d}'.format(
        log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = args.device

    # load data
    data = DataLoader(args, device, logging)

    # construct model & optimizer
    model = TransH(args, data.n_entities, data.n_relations, device)

    print("Device {}".format(device))
    model.to(device)

    logging.info(model)
    torch.autograd.set_detect_anomaly(True)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    # train
    loss_kg_list = []

    kg_time_training = []

    pytorch_total_params = sum(p.numel()
                               for p in model.parameters() if p.requires_grad)
    print("Total parameters: {}".format(pytorch_total_params))


    # Train model
    for epoch in range(1, args.n_epoch + 1):
        model.train()
        time3 = time()
        kg_total_loss = 0

        # Sampling data for each epoch
        n_data_samples = int(len(list(data.train_kg_dict)) * args.epoch_data_rate)
        epoch_sampling_data_list = random.sample(list(data.train_kg_dict), n_data_samples)
        epoch_sampling_data_dict = {k: data.train_kg_dict[k] for k in epoch_sampling_data_list}
        n_kg_batch = n_data_samples // data.pre_training_batch_size + 1

        for iter in tqdm(range(1, n_kg_batch + 1), desc=f"EP:{epoch}_train"):
            time4 = time()
            kg_batch_head, kg_batch_relation, kg_batch_pos_tail, kg_batch_neg_tail = data.generate_kg_batch(
                epoch_sampling_data_dict, data.pre_training_batch_size, data.n_entities)
            kg_batch_head = kg_batch_head.to(device)
            kg_batch_relation = kg_batch_relation.to(device)
            kg_batch_pos_tail = kg_batch_pos_tail.to(device)
            kg_batch_neg_tail = kg_batch_neg_tail.to(device)

            optimizer.zero_grad()
            kg_batch_loss = model(kg_batch_head, kg_batch_relation,
                                  kg_batch_pos_tail, kg_batch_neg_tail)

            if np.isnan(kg_batch_loss.cpu().detach().numpy()):
                logging.info(
                    'ERROR (Training): Epoch {:04d} Iter {:04d} / {:04d} Loss is nan.'.format(epoch, iter,
                                                                                                  n_kg_batch))
                sys.exit()

            kg_batch_loss.backward()
            optimizer.step()
            kg_total_loss += kg_batch_loss.item()

            if iter % 50 == 0:
                torch.cuda.empty_cache()

            loss_value = kg_total_loss / n_kg_batch

            if (iter % args.kg_print_every) == 0:
                logging.info(
                    'Training: Epoch {:04d}/{:04d} Iter {:04d} / {:04d} | Time {:.1f}s | Iter Loss {:.4f} | Iter Mean Loss {:.4f}'.format(
                        epoch, args.n_epoch, iter, n_kg_batch, time() - time4, kg_batch_loss.item(),
                                                               kg_total_loss / iter))
        logging.info(
            'Pre-training: Epoch {:04d}/{:04d} Total Iter {:04d} | Total Time {:.1f}s | Iter Mean Loss {:.4f}'.format(
                epoch, args.n_epoch, n_kg_batch, time() - time3, loss_value))

        loss_kg_list.append(loss_value)
        kg_time_training.append(time() - time3)

        torch.cuda.empty_cache()

        # Logging every epoch
        logging.info("Loss KG {}".format(loss_kg_list))
        logging.info("Training KG time {}".format(kg_time_training))
    
    logging.info("FINALLL -------")
    # Logging every epoch
    logging.info("KG loss list {}".format(loss_kg_list))
    logging.info("KG time training {}".format(kg_time_training))


def parse_args():
    parser = argparse.ArgumentParser(description="Run KGAT.")

    parser.add_argument('--exp_name', type=str, default="run")
    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')

    parser.add_argument('--data_name', nargs='?', default='Test',
                        help='Choose a dataset')
    parser.add_argument('--data_dir', nargs='?', default='data/',
                        help='Input data path.')

    parser.add_argument('--use_pretrain', type=int, default=0,
                        help='0: No pretrain, 1: Pretrain with the learned embeddings, 2: Pretrain with stored model.')
    parser.add_argument('--pretrain_embedding_dir', nargs='?', default='data/pretrain/',
                        help='Path of learned embeddings.')
    parser.add_argument('--pretrain_model_path', nargs='?', default='trained_model/model.pth',
                        help='Path of stored model.')

    parser.add_argument('--fine_tuning_batch_size', type=int, default=100,
                        help='Fine Tuning batch size.')
    parser.add_argument('--pre_training_batch_size', type=int, default=512,
                        help='KG batch size.')
    parser.add_argument('--test_batch_size', type=int, default=100,
                        help='Test batch size (the head number to test every batch).')

    parser.add_argument('--total_ent', type=int, default=1000,
                        help='Total entities.')
    parser.add_argument('--total_rel', type=int, default=100,
                        help='Total relations.')

    parser.add_argument('--embed_dim', type=int, default=300,
                        help='head / entity Embedding size.')
    parser.add_argument('--relation_dim', type=int, default=300,
                        help='Relation Embedding size.')
    parser.add_argument('--num_lit_dim', type=int, default=3,
                        help='Numerical Literal Embedding size.')
    parser.add_argument('--txt_lit_dim', type=int, default=300,
                        help='Text Literal Embedding size.')

    parser.add_argument('--laplacian_type', type=str, default='random-walk',
                        help='Specify the type of the adjacency (laplacian) matrix from {symmetric, random-walk}.')
    parser.add_argument('--aggregation_type', type=str, default='bi-interaction',
                        help='Specify the type of the aggregation layer from {gcn, graphsage, bi-interaction}.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')
    parser.add_argument('--mess_dropout', nargs='?', default='[0.1, 0.1, 0.1]',
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')

    parser.add_argument('--kg_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating KG l2 loss.')
    parser.add_argument('--fine_tuning_l2loss_lambda', type=float, default=1e-5,
                        help='Lambda when calculating Fine Tuning l2 loss.')

    parser.add_argument('--lr', type=float, default=0.0001,
                        help='Learning rate.')
    parser.add_argument('--n_epoch', type=int, default=50,
                        help='Number of epoch.')
    parser.add_argument('--epoch_data_rate', type=float, default=0.1,
                        help='Sampling data rate for each epoch.')
    parser.add_argument('--stopping_steps', type=int, default=10,
                        help='Number of epoch for early stopping')

    parser.add_argument('--fine_tuning_print_every', type=int, default=500,
                        help='Iter interval of printing Fine Tuning loss.')
    parser.add_argument('--kg_print_every', type=int, default=500,
                        help='Iter interval of printing KG loss.')
    parser.add_argument('--evaluate_every', type=int, default=20,
                        help='Epoch interval of evaluating Fine Tuning.')

    parser.add_argument('--pre_training_neg_rate', type=int, default=3,
                        help='The pre-training negative rate.')

    parser.add_argument('--device', nargs='?', default='cpu',
                        help='Choose a device to run')
    parser.add_argument('--prediction_dict_file', nargs='?', default='disease_dict.pickle',
                        help='Disease dictionary file')

    parser.add_argument('--use_residual', type=bool, default=True,
                        help='Use residual connection.')

    parser.add_argument('--use_parallel_gpu', type=bool, default=True,
                        help='Use many GPUs.')

    parser.add_argument('--alpha', type=float, default=0.1, help='alpha_l')
    parser.add_argument('--lamda', type=float, default=0.5, help='lamda.')

    args = parser.parse_args()

    args.data_name = args.data_name.replace("'", "")

    save_dir = 'result'
    args.save_dir = save_dir

    return args

def fix_seed(random_seed):
    torch.manual_seed(random_seed)
    torch.cuda.manual_seed(random_seed)
    torch.cuda.manual_seed_all(random_seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(random_seed)
    random.seed(random_seed)

def main():
    args = parse_args()
    train(args)


if __name__ == '__main__':
    main()
