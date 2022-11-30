import os
import random
import collections

import torch
import numpy as np
import pandas as pd


class DataLoaderBase(object):

    def __init__(self, args, device, logging):
        self.args = args
        self.data_name = args.data_name
        self.use_pretrain = args.use_pretrain
        self.pretrain_embedding_dir = args.pretrain_embedding_dir
        self.device = device

        self.data_dir = os.path.join(args.data_dir, args.data_name)
        self.kg_file = os.path.join(self.data_dir, "data.txt")

        self.entity_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.total_ent = args.total_ent
        self.total_rel = args.total_rel

        self.pre_training_neg_rate = args.pre_training_neg_rate


    def load_graph(self, filename):
        graph_data = pd.read_csv(filename, sep=' ', names=[
            'h', 'r', 't'], engine='python')
        graph_data = graph_data.drop_duplicates()
        return graph_data

    def sample_pos_triples_for_head(self, kg_dict, head, n_sample_pos_triples):
        pos_triples = kg_dict[head]
        n_pos_triples = len(pos_triples)

        sample_relations, sample_pos_tails = [], []
        while True:
            if len(sample_relations) == n_sample_pos_triples:
                break

            pos_triple_idx = np.random.randint(
                low=0, high=n_pos_triples, size=1)[0]
            tail = pos_triples[pos_triple_idx][0]
            relation = pos_triples[pos_triple_idx][1]

            if relation not in sample_relations and tail not in sample_pos_tails:
                sample_relations.append(relation)
                sample_pos_tails.append(tail)
        return sample_relations, sample_pos_tails

    def sample_neg_triples_for_head(self, kg_dict, head, relation, n_sample_neg_triples, highest_neg_idx):
        pos_triples = kg_dict[head]

        sample_neg_tails = []
        while True:
            if len(sample_neg_tails) == n_sample_neg_triples:
                break

            tail = np.random.randint(low=0, high=highest_neg_idx, size=1)[0]
            if (tail, relation) not in pos_triples and tail not in sample_neg_tails:
                sample_neg_tails.append(tail)
        return sample_neg_tails

    def generate_kg_batch(self, kg_dict, batch_size, highest_neg_idx):
        exist_heads = kg_dict.keys()
        batch_size = int(batch_size / self.pre_training_neg_rate)

        if batch_size <= len(exist_heads):
            batch_head = random.sample(exist_heads, batch_size)
        else:
            batch_head = [random.choice(exist_heads)
                          for _ in range(batch_size)]

        batch_relation, batch_pos_tail, batch_neg_tail = [], [], []

        for h in batch_head:
            # Generate the positive samples
            relation, pos_tail = self.sample_pos_triples_for_head(
                kg_dict, h, 1)
            batch_relation += relation
            batch_pos_tail += pos_tail

            # Generate the negative samples
            neg_tail = self.sample_neg_triples_for_head(
                kg_dict, h, relation[0], self.pre_training_neg_rate, highest_neg_idx)

            batch_neg_tail += neg_tail

        batch_head = self.generate_batch_by_neg_rate(batch_head, self.pre_training_neg_rate)
        batch_relation = self.generate_batch_by_neg_rate(batch_relation, self.pre_training_neg_rate)
        batch_pos_tail = self.generate_batch_by_neg_rate(batch_pos_tail, self.pre_training_neg_rate)

        batch_head = torch.LongTensor(batch_head)
        batch_relation = torch.LongTensor(batch_relation)
        batch_pos_tail = torch.LongTensor(batch_pos_tail)
        batch_neg_tail = torch.LongTensor(batch_neg_tail)
        return batch_head, batch_relation, batch_pos_tail, batch_neg_tail

    def generate_batch_by_neg_rate(self, batch, rate):
        zip_list = []
        results = []

        for i in range(rate):
            zip_list.append(batch)

        zip_list = list(zip(*zip_list))

        for x in zip_list:
            results += list(x)

        return results


class DataLoader(DataLoaderBase):

    def __init__(self, args, device, logging):
        super().__init__(args, device, logging)
        self.pre_training_batch_size = int(args.pre_training_batch_size / self.pre_training_neg_rate)

        graph_data = self.load_graph(self.kg_file)
        self.construct_data(graph_data)
        self.print_info(logging)

    def construct_data(self, graph_data):
        # Removed addition of inverse

        # re-map head id
        # graph_data['r'] += 2
        self.n_heads = max(graph_data['h'])
        self.n_tails = max(graph_data['t']) 
        self.n_entities = max(self.n_heads, self.n_tails) + 1
        self.n_relations = len(set(graph_data['r']))

        self.pre_train_data = graph_data
        self.n_pre_training = len(self.pre_train_data)

        # construct kg dict
        h_list = []
        t_list = []
        r_list = []

        self.train_kg_dict = collections.defaultdict(list)
        self.train_relation_dict = collections.defaultdict(list)

        for row in self.pre_train_data.iterrows():
            h, r, t = row[1]

            h_list.append(h)
            t_list.append(t)
            r_list.append(r)

            self.train_kg_dict[h].append((t, r))
            self.train_relation_dict[r].append((h, t))

        self.h_list = torch.LongTensor(h_list)
        self.t_list = torch.LongTensor(t_list)
        self.r_list = torch.LongTensor(r_list)

    def print_info(self, logging):
        logging.info('Total training heads:           %d' % self.n_heads)
        logging.info('Total training tails:           %d' % self.n_tails)
        logging.info('Total entities:        %d' % self.n_entities)
        logging.info('n_relations:       %d' % self.n_relations)

        logging.info('n_h_list:          %d' % len(self.h_list))
        logging.info('n_t_list:          %d' % len(self.t_list))
        logging.info('n_r_list:          %d' % len(self.r_list))

        logging.info('n_pre_training:        %d' % self.n_pre_training)