import torch
import torch.nn as nn
import torch.nn.functional as F

def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)

class TransE(nn.Module):

    def __init__(self, args, n_entities, n_relations, device=None):

        super(TransE, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.training_neg_rate = args.training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        # Trans E
        pos_score = torch.sum(
            torch.pow(head_embed + r_embed - tail_pos_embed, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(head_embed + r_embed - tail_neg_embed, 2), dim=1)  # (kg_batch_size)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(
            tail_neg_embed)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)

class TransH(nn.Module):
    def __init__(self, args, n_entities, n_relations, device=None, p_norm=1, margin = None):
        super(TransH, self).__init__()
        self.device = device
        self.device = device
        self.n_entities = n_entities
        self.n_relations = n_relations
        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim
        self.p_norm = p_norm

        self.ent_embeddings = nn.Embedding(self.n_entities, self.embed_dim)
        self.rel_embeddings = nn.Embedding(self.n_relations, self.relation_dim)
        self.norm_vector = nn.Embedding(self.n_relations, self.relation_dim)

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.training_neg_rate = args.training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

        if margin != None:
            self.margin = nn.Parameter(torch.Tensor([margin]))
            self.margin.requires_grad = False
            self.margin_flag = True
        else:
            self.margin_flag = False

    def _transfer(self, e, norm):
        norm = F.normalize(norm, p = 2, dim = -1)
        if e.shape[0] != norm.shape[0]:
            e = e.view(-1, norm.shape[0], e.shape[-1])
            norm = norm.view(-1, norm.shape[0], norm.shape[-1])
            e = e - torch.sum(e * norm, -1, True) * norm
            return e.view(-1, e.shape[-1])
        else:
            return e - torch.sum(e * norm, -1, True) * norm

    def forward(self, h, r, pos_t, neg_t):
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)
        r_norm = self.norm_vector(r)

        head_embed_trans = self._transfer(head_embed, r_norm)
        tail_pos_embed_trans = self._transfer(tail_pos_embed, r_norm)
        tail_neg_embed_trans = self._transfer(tail_neg_embed, r_norm)
        pos_score = torch.sum(
            torch.pow(head_embed_trans + r_embed - tail_pos_embed_trans, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(head_embed_trans + r_embed - tail_neg_embed_trans, 2), dim=1)  # (kg_batch_size)

        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_embed_trans) + _L2_loss_mean(r_embed) + _L2_loss_mean(tail_pos_embed_trans) + _L2_loss_mean(
            tail_neg_embed_trans)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss
        return loss

class TransR(nn.Module):

    def __init__(self, args, n_entities, n_relations, device=None):

        super(TransR, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.training_neg_rate = args.training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)
        # self.trans_M = nn.Parameter(torch.Tensor(
        #     self.n_relations, self.embed_dim, self.relation_dim))
        self.trans_M = nn.Parameter(torch.Tensor(
            self.n_relations, self.embed_dim,
            self.relation_dim))

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)
        # nn.init.xavier_uniform_(self.trans_M)
        nn.init.xavier_uniform_(self.trans_M)

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)
        W_r = self.trans_M[r]  # (kg_batch_size, embed_dim, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        r_mul_h = torch.bmm(head_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_pos_t = torch.bmm(tail_pos_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)
        r_mul_neg_t = torch.bmm(tail_neg_embed.unsqueeze(1), W_r).squeeze(
            1)  # (kg_batch_size, relation_dim)

        # Trans R
        # Equation (1)
        pos_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_pos_t, 2), dim=1)  # (kg_batch_size)
        neg_score = torch.sum(
            torch.pow(r_mul_h + r_embed - r_mul_neg_t, 2), dim=1)  # (kg_batch_size)

        # Equation (2)
        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(r_mul_h) + _L2_loss_mean(r_embed) + _L2_loss_mean(r_mul_pos_t) + _L2_loss_mean(
            r_mul_neg_t)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss
        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)


class RESCAL(nn.Module):

    def __init__(self, args, n_entities, n_relations, device=None):

        super(RESCAL, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.training_neg_rate = args.training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim*self.embed_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

    def calculate_score(self, h, t, r):
        t = t.view(-1, self.embed_dim, 1)
        r = r.view(-1, self.relation_dim, self.embed_dim)
        tr = torch.matmul(r, t)
        tr = tr.view(-1, self.embed_dim)
        return -torch.sum(h * tr, -1)

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        # 
        pos_score = self.calculate_score(head_embed, tail_pos_embed, r_embed)
        neg_score = self.calculate_score(head_embed, tail_neg_embed, r_embed)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(
            tail_neg_embed)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss

        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)

class DistMult(nn.Module):

    def __init__(self, args, n_entities, n_relations, device=None):

        super(DistMult, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.training_neg_rate = args.training_neg_rate

        self.entity_embed = nn.Embedding(
            self.n_entities, self.embed_dim)
        self.relation_embed = nn.Embedding(self.n_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.entity_embed.weight)

        nn.init.xavier_uniform_(self.relation_embed.weight)

    def calculate_score(self, h, t, r):
        score = (h * r) * t
        score = torch.sum(score, -1) 
        return score 

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_embed = self.relation_embed(r)  # (kg_batch_size, relation_dim)

        all_embed = self.entity_embed.weight

        head_embed = all_embed[h]  # (batch_size, concat_dim)
        tail_pos_embed = all_embed[pos_t]  # (batch_size, concat_dim)
        tail_neg_embed = all_embed[neg_t]  # (batch_size, concat_dim)

        # 
        pos_score = self.calculate_score(head_embed, tail_pos_embed, r_embed)
        neg_score = self.calculate_score(head_embed, tail_neg_embed, r_embed)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_embed) + _L2_loss_mean(r_embed) + _L2_loss_mean(tail_pos_embed) + _L2_loss_mean(
            tail_neg_embed)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss

        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)


class ComplEx(nn.Module):

    def __init__(self, args, n_entities, n_relations, device=None):

        super(ComplEx, self).__init__()
        self.device = device

        self.n_entities = n_entities
        self.n_relations = n_relations

        self.embed_dim = args.embed_dim
        self.relation_dim = args.relation_dim

        self.kg_l2loss_lambda = args.kg_l2loss_lambda

        self.ent_re_embeddings = nn.Embedding(self.n_entities, self.embed_dim)
        self.ent_im_embeddings = nn.Embedding(self.n_entities, self.embed_dim)
        self.rel_re_embeddings = nn.Embedding(self.n_relations, self.relation_dim)
        self.rel_im_embeddings = nn.Embedding(self.n_relations, self.relation_dim)

        nn.init.xavier_uniform_(self.ent_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.ent_im_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_re_embeddings.weight.data)
        nn.init.xavier_uniform_(self.rel_im_embeddings.weight.data)

    def calculate_score(self, h_re, h_im, t_re, t_im, r_re, r_im):
        return torch.sum(
            h_re * t_re * r_re
            + h_im * t_im * r_re
            + h_re * t_im * r_im
            - h_im * t_re * r_im,
            -1
        )

    def calc_triplet_loss(self, h, r, pos_t, neg_t):
        """
        h:      (kg_batch_size)
        r:      (kg_batch_size)
        pos_t:  (kg_batch_size)
        neg_t:  (kg_batch_size)
        """
        r_re_embed = self.rel_re_embeddings(r)
        r_im_embed = self.rel_im_embeddings(r)

        ent_re_embed = self.ent_re_embeddings.weight
        ent_im_embed = self.ent_im_embeddings.weight

        head_re_embed = ent_re_embed[h]
        head_im_embed = ent_im_embed[h]

        tail_re_pos_embed = ent_re_embed[pos_t]
        tail_im_pos_embed = ent_im_embed[pos_t]

        tail_re_neg_embed = ent_re_embed[neg_t]
        tail_im_neg_embed = ent_im_embed[neg_t]

        # 
        pos_score = self.calculate_score(head_re_embed, head_im_embed, tail_re_pos_embed, tail_im_pos_embed, r_re_embed, r_im_embed)
        neg_score = self.calculate_score(head_re_embed, head_im_embed, tail_re_neg_embed, tail_im_neg_embed, r_re_embed, r_im_embed)

        # triplet_loss = F.softplus(pos_score - neg_score)
        triplet_loss = (-1.0) * F.logsigmoid(neg_score - pos_score)
        triplet_loss = torch.mean(triplet_loss)

        l2_loss = _L2_loss_mean(head_re_embed) + _L2_loss_mean(head_im_embed) + _L2_loss_mean(tail_re_pos_embed) + _L2_loss_mean(
            tail_im_pos_embed) + _L2_loss_mean(tail_re_neg_embed) + _L2_loss_mean(tail_im_neg_embed) + _L2_loss_mean(r_re_embed) + _L2_loss_mean(r_im_embed)
        loss = triplet_loss + self.kg_l2loss_lambda * l2_loss

        return loss

    def forward(self, *input):
        return self.calc_triplet_loss(*input)
