import numpy as np

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different
        (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) *
        (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))

def swish(x):
    return x * torch.sigmoid(x)

def wasserstein_distance(mean1, cov1, mean2, cov2):
    ret = torch.sum((mean1 - mean2) * (mean1 - mean2), -1)
    cov1_sqrt = torch.sqrt(torch.clamp(cov1, min=1e-24)) 
    cov2_sqrt = torch.sqrt(torch.clamp(cov2, min=1e-24))
    ret = ret + torch.sum((cov1_sqrt - cov2_sqrt) * (cov1_sqrt - cov2_sqrt), -1)

    return ret

def wasserstein_distance_matmul(mean1, cov1, mean2, cov2):
    mean1_2 = torch.sum(mean1**2, -1, keepdim=True)
    mean2_2 = torch.sum(mean2**2, -1, keepdim=True)
    ret = -2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2)
    #ret = torch.clamp(-2 * torch.matmul(mean1, mean2.transpose(-1, -2)) + mean1_2 + mean2_2.transpose(-1, -2), min=1e-24)
    #ret = torch.sqrt(ret)

    cov1_2 = torch.sum(cov1, -1, keepdim=True)
    cov2_2 = torch.sum(cov2, -1, keepdim=True)
    #cov_ret = torch.clamp(-2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2), min=1e-24)
    #cov_ret = torch.sqrt(cov_ret)
    cov_ret = -2 * torch.matmul(torch.sqrt(torch.clamp(cov1, min=1e-24)), torch.sqrt(torch.clamp(cov2, min=1e-24)).transpose(-1, -2)) + cov1_2 + cov2_2.transpose(-1, -2)

    return ret + cov_ret

def kl_distance(mean1, cov1, mean2, cov2):
    trace_part = torch.sum(cov1 / cov2, -1)
    mean_cov_part = torch.sum((mean2 - mean1) / cov2 * (mean2 - mean1), -1)
    determinant_part = torch.log(torch.prod(cov2, -1) / torch.prod(cov1, -1))

    return (trace_part + mean_cov_part - mean1.shape[1] + determinant_part) / 2

def kl_distance_matmul(mean1, cov1, mean2, cov2):
    cov1_det = 1 / torch.prod(cov1, -1, keepdim=True)
    cov2_det = torch.prod(cov2, -1, keepdim=True)
    log_det = torch.log(torch.matmul(cov1_det, cov2_det.transpose(-1, -2)))

    trace_sum = torch.matmul(1 / cov2, cov1.transpose(-1, -2))

    #mean_cov_part1 = torch.matmul(mean1 / cov2, mean1.transpose(-1, -2))
    #mean_cov_part1 = torch.matmul(mean1 * mean1, (1 / cov2).transpose(-1, -2))
    #mean_cov_part2 = -torch.matmul(mean1 / cov2, mean2.transpose(-1, -2))
    #mean_cov_part2 = -torch.matmul(mean1 * mean2, (1 / cov2).transpose(-1, -2))
    #mean_cov_part3 = -torch.matmul(mean2 / cov2, mean1.transpose(-1, -2))
    #mean_cov_part4 = torch.matmul(mean2 / cov2, mean2.transpose(-1, -2))
    #mean_cov_part4 = torch.matmul(mean2 * mean2, (1 / cov2).transpose(-1, -2))

    #mean_cov_part = mean_cov_part1 + mean_cov_part2 + mean_cov_part3 + mean_cov_part4
    mean_cov_part = torch.matmul((mean1 - mean2) ** 2, (1/cov2).transpose(-1, -2))

    return (log_det + mean_cov_part + trace_sum - mean1.shape[-1]) / 2


def d2s_gaussiannormal(distance):

    return torch.exp(-distance)

def d2s_1overx(distance):

    return 1/(1+distance)
    


ACT2FN = {"gelu": gelu, "relu": F.relu, "swish": swish}


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from item, position.
    """
    def __init__(self, args):
        super(Embeddings, self).__init__()

        self.item_embeddings = nn.Embedding(args.item_size, args.hidden_size, padding_idx=0) # 不要乱用padding_idx
        self.position_embeddings = nn.Embedding(args.max_seq_length, args.hidden_size)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)

        self.args = args

    def forward(self, input_ids):
        seq_length = input_ids.size(1)
        position_ids = torch.arange(seq_length, dtype=torch.long, device=input_ids.device)
        position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
        items_embeddings = self.item_embeddings(input_ids)
        position_embeddings = self.position_embeddings(position_ids)
        embeddings = items_embeddings + position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

class SelfAttention(nn.Module):
    def __init__(self, args):
        super(SelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_tensor, attention_mask):
        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        # [batch_size heads seq_len seq_len] scores
        # [batch_size 1 1 seq_len]
        attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        # Fixme
        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states

class RelationAwareSelfAttention(nn.Module):
    def __init__(self, args, num_relationships):
        super(RelationAwareSelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_relationships = num_relationships
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(args.hidden_size, self.all_head_size)
        self.key = nn.Linear(args.hidden_size, self.all_head_size)
        self.value = nn.Linear(args.hidden_size, self.all_head_size)

        self.relationship_embedding = nn.Parameter(torch.rand(self.num_relationships, args.hidden_size, self.all_head_size))
        self.relationship_weights = nn.Parameter(torch.rand(self.num_relationships))

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)

        self.dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self._init_rel_score_norm(args)

    # ------------------------------------------------------------------
    # Relation-score normalization
    #
    # MEASURED PROBLEM (hidden_size=128, 1 head, LayerNormed input):
    #   ordinary attention scores  ~ 1.7 in magnitude
    #   relation attention scores  ~ 45,000 in magnitude (std ~3,800)
    #
    # relationship_embedding is initialized as torch.rand(R, d, d) -- uniform
    # on [0, 1], every entry POSITIVE -- and then squared via bmm, so
    # relationship_embedding_sym has entries around d/4 = 32. Two einsum
    # contractions over d=128 later, the relation term dominates the ordinary
    # attention term by ~4 orders of magnitude.
    #
    # Consequence: softmax(attention_scores + rel_sum) is fully saturated at
    # initialization -- measured mean max attention probability 0.9987, and
    # attention entropy 0.003 out of a possible log(100) = 4.61. Gradients
    # through a saturated softmax vanish, so the ENTIRE attention block sees
    # ~1e-9 gradients, including relationship_weights (MT4SR's global relation
    # weighting) and relation_gate (D-MT4SR's context-conditioned gate). The
    # relation weighting the model is supposed to learn therefore stays frozen
    # at its random initialization.
    #
    # rel_score_norm puts the relation pathway back on the same numerical
    # footing as ordinary attention:
    #   'none'      original behavior (keep for exact reproduction of MT4SR)
    #   'std'       standardize relation scores per (batch, head, query) over
    #               the (key, relation) dims, then apply a learnable gain
    #   'layernorm' LayerNorm the relation MAPPING vectors before the score
    #               matmul, then apply a learnable gain -- normalizes the
    #               representation rather than the scores, so it does not
    #               interact with causal/padding masking at all
    # In both cases the gain is initialized to 1.0 and learned.
    #
    # This lives on the PARENT class deliberately: the fix must be available
    # to plain MT4SR too, so `baseline + rel_score_norm` can be run as a
    # control. Without that control there is no way to tell "dynamic gating
    # helps" apart from "we fixed a saturation bug and now any relation
    # weighting trains".
    # ------------------------------------------------------------------
    def _init_rel_score_norm(self, args):
        self.rel_score_norm = getattr(args, 'rel_score_norm', 'none')
        if self.rel_score_norm not in ('none', 'std', 'layernorm'):
            raise ValueError(f"Unknown rel_score_norm '{self.rel_score_norm}' "
                             "(expected none, std or layernorm)")
        if self.rel_score_norm != 'none':
            self.rel_score_gain = nn.Parameter(torch.ones(1))
        if self.rel_score_norm == 'layernorm':
            self.rel_map_LayerNorm = LayerNorm(self.all_head_size, eps=1e-12)
        # Diagnostics, refreshed every forward pass and read by the trainer so
        # the log alone shows whether the relation pathway is saturated.
        self.last_rel_score_absmax = None
        self.last_attn_entropy = None

    def normalize_rel_mapping(self, relationship_mapping):
        """Applied to (B, num_rel, L, d) before the relation score matmul."""
        if self.rel_score_norm == 'layernorm':
            return self.rel_map_LayerNorm(relationship_mapping)
        return relationship_mapping

    def normalize_rel_scores(self, scores):
        """Applied to (B, h, L, L, num_rel) relation scores."""
        if self.rel_score_norm == 'std':
            mean = scores.mean(dim=(-2, -1), keepdim=True)
            std = scores.std(dim=(-2, -1), keepdim=True)
            scores = (scores - mean) / (std + 1e-6)
        if self.rel_score_norm != 'none':
            scores = scores * self.rel_score_gain
        return scores

    @torch.no_grad()
    def record_attention_diagnostics(self, rel_scores, attention_probs):
        """Caches saturation diagnostics for the trainer to log."""
        self.last_rel_score_absmax = float(rel_scores.detach().abs().max())
        p = attention_probs.detach().clamp_min(1e-12)
        self.last_attn_entropy = float(-(p * p.log()).sum(-1).mean())

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)
    
    def transpose_for_scores_relation(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size) # (B, num_rel, L, h, d/h)
        x = x.view(*new_x_shape)
        return x.permute(0, 1, 3, 2, 4)

    def forward(self, input_tensor, attention_mask, input_relationships_masks):
        #relationship_embedding shape: [num_rel, d, d]
        #relationship_weights shape: [num_rel]
        #input_relationships_masks shape: [num_rel, L, L]

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        #attention_probs = nn.Softmax(dim=-1)(attention_scores)


        relationship_weight_prob = nn.Softmax(dim=0)(self.relationship_weights)

        relationship_embedding_sym = torch.bmm(self.relationship_embedding, self.relationship_embedding)

        #relationship_mapping = torch.einsum("ijk,ltk->ijlt", (input_tensor, self.relationship_embedding)) # get (B, L, num_rel, d)
        relationship_mapping = torch.einsum("ijk,ltk->ijlt", (input_tensor, relationship_embedding_sym)) # get (B, L, num_rel, d)
        relationship_mapping = relationship_mapping.permute(0, 2, 1, 3).contiguous() # get (B, num_rel, L, d)
        # No-op unless --rel_score_norm=layernorm (see _init_rel_score_norm).
        relationship_mapping = self.normalize_rel_mapping(relationship_mapping)
        relationship_heads_mapping = self.transpose_for_scores_relation(relationship_mapping) # get (B, num_rel, h, L, d/h)
        relationship_att_scores = torch.matmul(relationship_heads_mapping, relationship_heads_mapping.transpose(-1, -2)) # get(B, num_rel, h, L, L)
        relationship_att_scores = relationship_att_scores / math.sqrt(self.attention_head_size)
        expanded_input_relationships_masks = input_relationships_masks.unsqueeze(2).expand(-1, -1, self.num_attention_heads, -1, -1)
        #relationship_att_scores = relationship_att_scores * expanded_input_relationships_masks / math.sqrt(self.attention_head_size)
        relationship_att_scores = relationship_att_scores / math.sqrt(self.attention_head_size)
        relationship_att_scores = relationship_att_scores.permute(0, 2, 3, 4, 1).contiguous()
        # No-op unless --rel_score_norm is set. Without it these scores are
        # ~4 orders of magnitude larger than attention_scores and saturate the
        # softmax below, killing gradients for the whole block.
        relationship_att_scores = self.normalize_rel_scores(relationship_att_scores)

        rel_sum_relationship_att_scores = torch.matmul(relationship_att_scores, relationship_weight_prob).squeeze(-1)

        attention_probs = nn.Softmax(dim=-1)(attention_scores.clone() + rel_sum_relationship_att_scores)
        self.record_attention_diagnostics(rel_sum_relationship_att_scores, attention_probs)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistSelfAttention(nn.Module):
    def __init__(self, args):
        super(DistSelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(args.hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.mean_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.cov_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.distance_metric = args.distance_metric
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, attention_mask):
        mixed_mean_query_layer = self.mean_query(input_mean_tensor)
        mixed_mean_key_layer = self.mean_key(input_mean_tensor)
        mixed_mean_value_layer = self.mean_value(input_mean_tensor)

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor)) + 1
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor)) + 1
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor)) + 1

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        #if self.distance_metric == 'wasserstein':
        #    attention_scores = d2s_gaussiannormal(wasserstein_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #else:
        #    attention_scores = d2s_gaussiannormal(kl_distance(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        #attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        if self.distance_metric == 'wasserstein':
            attention_scores = d2s_gaussiannormal(wasserstein_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))
        else:
            attention_scores = d2s_gaussiannormal(kl_distance_matmul(mean_query_layer, cov_query_layer, mean_key_layer, cov_key_layer))

        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        attention_probs = self.attn_dropout(attention_probs)
        mean_context_layer = torch.matmul(attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(attention_probs ** 2, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states, attention_probs


class DistMeanSelfAttention(nn.Module):
    def __init__(self, args):
        super(DistMeanSelfAttention, self).__init__()
        if args.hidden_size % args.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (args.hidden_size, args.num_attention_heads))
        self.num_attention_heads = args.num_attention_heads
        self.attention_head_size = int(args.hidden_size / args.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.mean_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.mean_value = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_key = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_query = nn.Linear(args.hidden_size, self.all_head_size)
        self.cov_value = nn.Linear(args.hidden_size, self.all_head_size)

        self.activation = nn.ELU()

        self.attn_dropout = nn.Dropout(args.attention_probs_dropout_prob)
        self.mean_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.cov_dense = nn.Linear(args.hidden_size, args.hidden_size)
        self.out_dropout = nn.Dropout(args.hidden_dropout_prob)

        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)


    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, input_mean_tensor, input_cov_tensor, attention_mask):
        mixed_mean_query_layer = self.mean_query(input_mean_tensor)
        mixed_mean_key_layer = self.mean_key(input_mean_tensor)
        mixed_mean_value_layer = self.mean_value(input_mean_tensor)

        mean_query_layer = self.transpose_for_scores(mixed_mean_query_layer)
        mean_key_layer = self.transpose_for_scores(mixed_mean_key_layer)
        mean_value_layer = self.transpose_for_scores(mixed_mean_value_layer)

        mixed_cov_query_layer = self.activation(self.cov_query(input_cov_tensor)) + 1
        mixed_cov_key_layer = self.activation(self.cov_key(input_cov_tensor)) + 1
        mixed_cov_value_layer = self.activation(self.cov_value(input_cov_tensor)) + 1

        cov_query_layer = self.transpose_for_scores(mixed_cov_query_layer)
        cov_key_layer = self.transpose_for_scores(mixed_cov_key_layer)
        cov_value_layer = self.transpose_for_scores(mixed_cov_value_layer)

        mean_attention_scores = torch.matmul(mean_query_layer, mean_key_layer.transpose(-1, -2))
        cov_attention_scores = torch.matmul(cov_query_layer, cov_key_layer.transpose(-1, -2))

        mean_attention_scores = mean_attention_scores / math.sqrt(self.attention_head_size)
        mean_attention_scores = mean_attention_scores + attention_mask
        mean_attention_probs = nn.Softmax(dim=-1)(mean_attention_scores)

        cov_attention_scores = cov_attention_scores / math.sqrt(self.attention_head_size)
        cov_attention_scores = cov_attention_scores + attention_mask
        cov_attention_probs = nn.Softmax(dim=-1)(cov_attention_scores)

        mean_attention_probs = self.attn_dropout(mean_attention_probs)
        cov_attention_probs = self.attn_dropout(cov_attention_probs)
        mean_context_layer = torch.matmul(mean_attention_probs, mean_value_layer)
        cov_context_layer = torch.matmul(cov_attention_probs, cov_value_layer)
        mean_context_layer = mean_context_layer.permute(0, 2, 1, 3).contiguous()
        cov_context_layer = cov_context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = mean_context_layer.size()[:-2] + (self.all_head_size,)

        mean_context_layer = mean_context_layer.view(*new_context_layer_shape)
        cov_context_layer = cov_context_layer.view(*new_context_layer_shape)

        mean_hidden_states = self.mean_dense(mean_context_layer)
        mean_hidden_states = self.out_dropout(mean_hidden_states)
        mean_hidden_states = self.LayerNorm(mean_hidden_states + input_mean_tensor)

        cov_hidden_states = self.cov_dense(cov_context_layer)
        cov_hidden_states = self.out_dropout(cov_hidden_states)
        cov_hidden_states = self.LayerNorm(cov_hidden_states + input_cov_tensor)

        return mean_hidden_states, cov_hidden_states, mean_attention_probs



class Intermediate(nn.Module):
    def __init__(self, args):
        super(Intermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        if isinstance(args.hidden_act, str):
            self.intermediate_act_fn = ACT2FN[args.hidden_act]
        else:
            self.intermediate_act_fn = args.hidden_act

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DistIntermediate(nn.Module):
    def __init__(self, args):
        super(DistIntermediate, self).__init__()
        self.dense_1 = nn.Linear(args.hidden_size, args.hidden_size * 4)
        self.intermediate_act_fn = nn.ELU()

        self.dense_2 = nn.Linear(args.hidden_size * 4, args.hidden_size)
        self.LayerNorm = LayerNorm(args.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(args.hidden_dropout_prob)


    def forward(self, input_tensor):

        hidden_states = self.dense_1(input_tensor)
        hidden_states = self.intermediate_act_fn(hidden_states)

        hidden_states = self.dense_2(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class Layer(nn.Module):
    def __init__(self, args):
        super(Layer, self).__init__()
        self.attention = SelfAttention(args)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask):
        attention_output = self.attention(hidden_states, attention_mask)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class RelationAwareLayer(nn.Module):
    def __init__(self, args, num_relationships):
        super(RelationAwareLayer, self).__init__()
        self.attention = RelationAwareSelfAttention(args, num_relationships)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask, input_relationships_masks):
        attention_output = self.attention(hidden_states, attention_mask, input_relationships_masks)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class DistLayer(nn.Module):
    def __init__(self, args):
        super(DistLayer, self).__init__()
        self.attention = DistSelfAttention(args)
        self.mean_intermediate = DistIntermediate(args)
        self.cov_intermediate = DistIntermediate(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask):
        mean_attention_output, cov_attention_output, attention_scores = self.attention(mean_hidden_states, cov_hidden_states, attention_mask)
        mean_intermediate_output = self.mean_intermediate(mean_attention_output)
        cov_intermediate_output = self.activation_func(self.cov_intermediate(cov_attention_output)) + 1
        return mean_intermediate_output, cov_intermediate_output, attention_scores


class DistMeanSALayer(nn.Module):
    def __init__(self, args):
        super(DistMeanSALayer, self).__init__()
        self.attention = DistMeanSelfAttention(args)
        self.mean_intermediate = DistIntermediate(args)
        self.cov_intermediate = DistIntermediate(args)
        self.activation_func = nn.ELU()

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask):
        mean_attention_output, cov_attention_output, attention_scores = self.attention(mean_hidden_states, cov_hidden_states, attention_mask)
        mean_intermediate_output = self.mean_intermediate(mean_attention_output)
        cov_intermediate_output = self.activation_func(self.cov_intermediate(cov_attention_output)) + 1
        return mean_intermediate_output, cov_intermediate_output, attention_scores


class DistSAEncoder(nn.Module):               
    def __init__(self, args):
        super(DistSAEncoder, self).__init__()
        layer = DistLayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            maen_hidden_states, cov_hidden_states, att_scores = layer_module(mean_hidden_states, cov_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        return all_encoder_layers


class DistMeanSAEncoder(nn.Module):
    def __init__(self, args):
        super(DistMeanSAEncoder, self).__init__()
        layer = DistMeanSALayer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, mean_hidden_states, cov_hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            maen_hidden_states, cov_hidden_states, att_scores = layer_module(mean_hidden_states, cov_hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        if not output_all_encoded_layers:
            all_encoder_layers.append([mean_hidden_states, cov_hidden_states, att_scores])
        return all_encoder_layers


class RelationAwareSAEncoder(nn.Module):
    def __init__(self, args, num_relationships):
        super(RelationAwareSAEncoder, self).__init__()
        layer = RelationAwareLayer(args, num_relationships)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    #def forward(self, hidden_states, relationship_embedding, relationship_weights, attention_mask, input_relationships_masks, output_all_encoded_layers=True):
    def forward(self, hidden_states, attention_mask, input_relationships_masks, output_all_encoded_layers=True):
        all_encoder_layers = []
        relation_embs_all_layers = []
        relation_weights_all_layers = []

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, input_relationships_masks)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                relation_embs_all_layers.append(layer_module.attention.relationship_embedding)
                relation_weights_all_layers.append(layer_module.attention.relationship_weights)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            relation_embs_all_layers.append(self.layer[-1].attention.relationship_embedding)
            relation_weights_all_layers.append(self.layer[-1].attention.relationship_weights)

        return all_encoder_layers, relation_embs_all_layers, relation_weights_all_layers



# ---------------------------------------------------------------------------
# D-MT4SR additions: dynamic (context-conditioned, time-decayed) relation weighting
#
# In the original MT4SR (RelationAwareSelfAttention above), each relationship r
# contributes to attention via a *global* learnable scalar `relationship_weights[r]`,
# softmax-normalized once and shared across every user, sequence position, and
# batch for the entire dataset. This means the model can never express, e.g.,
# "this user's next-item intent leans more on 'co-searched' than 'similar brand'"
# or "this relation matters less the further back the related item was" -- both
# of which the MT4SR paper explicitly motivates but the architecture can't act on.
#
# DynamicRelationAwareSelfAttention replaces the fixed relation-weight vector
# with a small gating network conditioned on the current hidden states, so the
# relative importance of each relationship is computed per-sequence-position
# instead of once globally. It optionally also applies a learnable per-relationship
# decay as a function of the gap between two items in the sequence:
#   - if real interaction timestamps are available (preprocess_fromscratch.py now
#     saves them, and args.has_real_timestamps is set accordingly in main.py),
#     the decay uses the actual elapsed time between the two interactions;
#   - otherwise (older preprocessed .npy files that predate timestamp support,
#     or --use_time_decay without regenerating the data) it automatically falls
#     back to sequence-position distance as a practical proxy, exactly as before.
# This fallback is decided once at construction time from args, so behavior for
# a given run is deterministic and doesn't depend on the contents of any one
# batch.
#
# Everything else (Q/K/V projections, the relationship_embedding "normal matrix"
# trick from ANALOGY, causal masking, dropout, residual + LayerNorm) is identical
# to RelationAwareSelfAttention, so this is a drop-in replacement with the same
# core forward() signature (input_times is an optional extra arg) -- it can be
# swapped in via --model_name without touching the rest of the pipeline, and old
# models/data are completely unaffected.
# ---------------------------------------------------------------------------

class DynamicRelationAwareSelfAttention(RelationAwareSelfAttention):
    def __init__(self, args, num_relationships):
        super(DynamicRelationAwareSelfAttention, self).__init__(args, num_relationships)

        # Context-conditioned relation gate: replaces the single global
        # `relationship_weights` vector with a per-position distribution over
        # relationships, predicted from the current hidden state at each
        # sequence position (i.e., dynamic per user/sequence/timestep instead
        # of fixed for the whole dataset).
        #
        # --- D-MT4SR v2 gate options (all opt-in, all default off) ----------
        # gate_residual : anchor the gate to MT4SR's static relationship_weights
        #                 so gate_scale=0 reproduces MT4SR exactly.
        # gate_per_head : let each attention head have its own relation
        #                 distribution instead of sharing one across heads.
        # gate_pairwise : condition on the (query, key) PAIR via a low-rank
        #                 additive term, not just the query position.
        # gate_use_rel_mask : feed the observed also_buy/also_view pair mask
        #                 into the gate logits (this signal is otherwise
        #                 computed and thrown away).
        # gate_temperature / gate_entropy_weight : sharpen or regularize the
        #                 relation distribution to stop it collapsing early.
        # --------------------------------------------------------------------
        self.gate_residual = getattr(args, 'gate_residual', False)
        self.gate_per_head = getattr(args, 'gate_per_head', False)
        self.gate_pairwise = getattr(args, 'gate_pairwise', False)
        self.gate_use_rel_mask = getattr(args, 'gate_use_rel_mask', False)
        self.gate_temperature = float(getattr(args, 'gate_temperature', 1.0))
        self.gate_entropy_weight = float(getattr(args, 'gate_entropy_weight', 0.0))

        # Number of independent relation distributions produced per position:
        # one per head if gate_per_head, otherwise one shared across heads.
        self.gate_groups = self.num_attention_heads if self.gate_per_head else 1
        gate_out = self.gate_groups * num_relationships

        self.relation_gate = nn.Linear(args.hidden_size, gate_out)

        if self.gate_pairwise:
            # Low-rank additive pairwise term: logits_ij = W_q h_i + W_k h_j.
            # Materializes two (B, L, groups*R) tensors and broadcast-adds them
            # to (B, L, L, groups*R) -- no new large parameter matrices, and no
            # tensor bigger than relationship_att_scores, which already exists.
            # Initialized to zero so a pairwise run starts exactly where the
            # query-only gate starts and only becomes pair-dependent if the
            # data supports it.
            self.relation_gate_key = nn.Linear(args.hidden_size, gate_out)
            nn.init.zeros_(self.relation_gate_key.weight)
            nn.init.zeros_(self.relation_gate_key.bias)

        if self.gate_use_rel_mask:
            # One learnable logit bonus per relationship, added to the gate
            # logits at exactly those (i, j) pairs where that relationship
            # actually holds in the item graph. Initialized to 0 so the run
            # starts identical to the mask-free gate; a positive learned value
            # means "trust this relation more where it is actually observed".
            self.relation_mask_bias = nn.Parameter(torch.zeros(num_relationships))

        if self.gate_residual:
            # Multiplicative scale on the context-dependent part of the gate.
            # Initialized to 0, so at step 0 the relation distribution is
            # exactly softmax(relationship_weights) -- i.e. MT4SR. The dynamic
            # component can only grow if it earns its keep. This makes
            # D-MT4SR a strict generalization of MT4SR rather than a
            # replacement, and removes the random-init relation prior that
            # made the plain dynamic gate so seed-sensitive.
            self.gate_scale = nn.Parameter(
                torch.full((1,), float(getattr(args, 'gate_scale_init', 0.0))))

        # Set by forward(); read by the trainer to add the entropy regularizer
        # to the loss. Kept as a plain attribute (not a buffer) so it never
        # ends up in state_dict and can't break checkpoint loading.
        self.last_gate_entropy = None

        # `relationship_weights` is inherited from the parent class. Without
        # --gate_residual it is unused (the gate replaces it outright, as
        # before); with --gate_residual it becomes the static anchor that the
        # gate perturbs.

        self.use_time_decay = getattr(args, 'use_time_decay', False)
        # Set once per run in main.py based on whether the loaded .npy file
        # actually contains timestamps (utils.get_user_seqs_MoHRdata). This is
        # a static, deterministic switch -- NOT re-checked per batch -- so a
        # given run always uses the same decay signal throughout training.
        self.has_real_timestamps = getattr(args, 'has_real_timestamps', False)
        if self.use_time_decay:
            # One learnable decay rate per relationship. Larger rate = relation's
            # influence falls off faster with distance (position or elapsed time).
            # Initialized small (slow decay) so the model starts close to the
            # non-decayed behavior and can learn stronger decay if useful.
            #
            # CALIBRATION WARNING: for sparse/infrequent-purchase categories
            # (e.g. Amazon Beauty), consecutive interactions can be months or
            # years apart. With the default time_scale (86400 = 1 day),
            # decay_rate=0.01 already suppresses ~87% of the relation signal
            # for a 200-day gap AT INITIALIZATION -- before training has a
            # chance to correct it. If a run trains for many more epochs than
            # its no-decay counterpart and underperforms, check the logged
            # decay_rate values (see trainers.DynamicRelationAwareSASRecModelTrainer);
            # near-zero decayed weights mean the relation signal has
            # effectively collapsed. Consider a larger --time_scale (e.g. a
            # month/year) for sparse categories.
            self.decay_rate = nn.Parameter(torch.ones(num_relationships) * 0.01)
            # Floor on the decay multiplier: even at maximum learned decay, at
            # least this fraction of the relation attention signal survives.
            # This bounds the worst case above -- decay_rate can still be
            # driven arbitrarily large by gradient descent, but it can no
            # longer fully zero out the relation pathway for distant pairs,
            # which is what caused HIT@5 to collapse (0.44 -> 0.27) in an
            # earlier ablation on All_Beauty. Default 0.1 keeps a floor of 10%.
            self.time_decay_floor = getattr(args, 'time_decay_floor', 0.1)
            if self.has_real_timestamps:
                # Divides raw timestamp gaps (typically unix seconds) down to a
                # sane scale before multiplying by decay_rate, so decay_rate
                # stays in an easily-learnable range regardless of the dataset's
                # time unit. Default 86400 = seconds/day, i.e. decay_rate is
                # roughly "decay per day".
                self.time_scale = getattr(args, 'time_scale', 86400.0)
            # Log-compressed gaps. Amazon review data spans ~20 years, so with
            # a 1-day time_scale the raw gap between two interactions routinely
            # reaches 10^3-10^4 "days" -- exp(-rate * gap) then saturates at the
            # floor for essentially every distant pair, which is why the floor
            # was needed at all and why every distant pair ends up with the
            # SAME weight (uninformative). Using log1p(gap) keeps the decay
            # monotone in elapsed time while staying in a range where a
            # learnable rate can actually discriminate a 1-week gap from a
            # 1-year gap.
            self.time_decay_log = getattr(args, 'time_decay_log', False)

    def forward(self, input_tensor, attention_mask, input_relationships_masks, input_times=None):
        # relationship_embedding shape: [num_rel, d, d]
        # input_relationships_masks shape: [num_rel, L, L]
        # input_times (optional): [B, L] raw per-position timestamps, aligned
        # with input_tensor's sequence positions (see datasets.py extra_tensors).
        # Only consulted when self.use_time_decay and self.has_real_timestamps
        # are both True; otherwise ignored (position-distance decay is used, or
        # no decay at all).

        mixed_query_layer = self.query(input_tensor)
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_scores = attention_scores + attention_mask

        relationship_embedding_sym = torch.bmm(self.relationship_embedding, self.relationship_embedding)

        relationship_mapping = torch.einsum("ijk,ltk->ijlt", (input_tensor, relationship_embedding_sym))  # (B, L, num_rel, d)
        relationship_mapping = relationship_mapping.permute(0, 2, 1, 3).contiguous()  # (B, num_rel, L, d)
        # No-op unless --rel_score_norm=layernorm.
        relationship_mapping = self.normalize_rel_mapping(relationship_mapping)
        relationship_heads_mapping = self.transpose_for_scores_relation(relationship_mapping)  # (B, num_rel, h, L, d/h)
        relationship_att_scores = torch.matmul(relationship_heads_mapping, relationship_heads_mapping.transpose(-1, -2))  # (B, num_rel, h, L, L)
        relationship_att_scores = relationship_att_scores / math.sqrt(self.attention_head_size)
        # NOTE: matches RelationAwareSelfAttention -- input_relationships_masks is
        # computed/available but not applied here either, to keep this an
        # apples-to-apples ablation of *only* the dynamic-weighting change.
        relationship_att_scores = relationship_att_scores.permute(0, 2, 3, 4, 1).contiguous()  # (B, h, L, L, num_rel)
        # No-op unless --rel_score_norm is set. Critical for the gate: without
        # it these scores saturate the attention softmax and relation_gate
        # receives ~1e-9 gradients, i.e. the "dynamic" relation weighting is a
        # frozen random projection rather than something the model learns.
        relationship_att_scores = self.normalize_rel_scores(relationship_att_scores)

        batch_size, seq_len = input_tensor.size(0), input_tensor.size(1)

        # --- Dynamic, context-conditioned relation weighting ---
        # Target broadcast shape is (B, h, L, L, num_rel), matching
        # relationship_att_scores. We build the gate logits at the smallest
        # rank that the enabled options require and let broadcasting do the
        # rest, so the query-only gate still costs (B, L, R) and only
        # --gate_pairwise / --gate_use_rel_mask pay for an (L, L) grid.
        G, R = self.gate_groups, self.num_relationships

        # (B, L, G, R): a distribution over relationships predicted for each
        # query position (and each head, if --gate_per_head) from its current
        # representation, instead of one fixed vector for the whole dataset.
        gate_logits = self.relation_gate(input_tensor).view(batch_size, seq_len, G, R)
        # -> (B, G, L, 1, R): [batch, head, query, key, relation]
        gate_logits = gate_logits.permute(0, 2, 1, 3).unsqueeze(3)

        if self.gate_pairwise:
            # Additive low-rank pairwise term: logit_ij = W_q h_i + W_k h_j, so
            # "which relation matters" becomes a property of the item PAIR
            # rather than of the query item alone.
            key_logits = self.relation_gate_key(input_tensor).view(batch_size, seq_len, G, R)
            key_logits = key_logits.permute(0, 2, 1, 3).unsqueeze(2)  # (B, G, 1, L, R)
            gate_logits = gate_logits + key_logits                    # (B, G, L, L, R)

        if self.gate_use_rel_mask and input_relationships_masks is not None:
            # input_relationships_masks: (B, num_rel, L, L), 1 where that
            # relationship holds between positions i and j. Give the gate a
            # learnable per-relationship bonus exactly on the observed edges.
            # This is the one genuinely new signal here: the masks are
            # computed by the dataset and, in the original MT4SR attention,
            # never actually consulted.
            rel_mask = input_relationships_masks.permute(0, 2, 3, 1)   # (B, L, L, R)
            rel_mask = rel_mask.unsqueeze(1)                           # (B, 1, L, L, R)
            gate_logits = gate_logits + rel_mask * self.relation_mask_bias.view(1, 1, 1, 1, R)

        if self.gate_residual:
            # Anchor on MT4SR's static per-relationship weights; the
            # context-dependent part is a scaled perturbation of them.
            # gate_scale == 0  =>  softmax(relationship_weights)  =>  MT4SR.
            gate_logits = (self.relationship_weights.view(1, 1, 1, 1, R)
                           + self.gate_scale * gate_logits)

        if self.gate_temperature != 1.0:
            gate_logits = gate_logits / self.gate_temperature

        dynamic_weights = nn.Softmax(dim=-1)(gate_logits)

        if self.gate_entropy_weight > 0.0 and self.training:
            # Mean entropy of the relation distribution over *valid* query
            # positions only. The trainer subtracts gate_entropy_weight * H
            # from the loss, i.e. rewards keeping the distribution spread out
            # early instead of collapsing onto one relationship, which is the
            # failure mode behind the large seed-to-seed spread.
            # attention_mask is 0 where a key is attendable and -10000 where it
            # is masked, so a query position is real iff any key is attendable.
            valid = (attention_mask > -1.0).any(dim=-1)               # (B, 1, L)
            valid = valid.view(batch_size, 1, seq_len, 1).float()     # (B,1,L,1)
            ent = -(dynamic_weights.clamp_min(1e-9).log() * dynamic_weights).sum(-1)
            # ent: (B, G, L, 1) or (B, G, L, L) -> average over key dim first
            ent = ent.mean(dim=-1, keepdim=True)
            denom = valid.sum() * G
            self.last_gate_entropy = (ent * valid).sum() / denom.clamp_min(1.0)
        else:
            self.last_gate_entropy = None

        if self.use_time_decay:
            decay_rate = torch.clamp(self.decay_rate, min=1e-4)  # (num_rel,), keep decay well-behaved
            floor = self.time_decay_floor
            if self.has_real_timestamps and input_times is not None:
                # Real elapsed-time decay: per-sample, per-position-pair gap in
                # (typically) days, since each sample has its own timestamps.
                time_gap = (input_times.unsqueeze(2) - input_times.unsqueeze(1)).abs() / self.time_scale  # (B, L, L)
                if getattr(self, 'time_decay_log', False):
                    # log1p compression: keeps decay monotone in elapsed time
                    # but stops multi-year gaps from saturating every distant
                    # pair to the floor (see __init__).
                    time_gap = torch.log1p(time_gap)
                raw_decay = torch.exp(-decay_rate.view(1, 1, 1, -1) * time_gap.unsqueeze(-1))  # (B, L, L, num_rel)
                # Floor: even for maximally-distant/old pairs, at least `floor`
                # of the relation signal survives, so a miscalibrated
                # time_scale/decay_rate can slow the model down but can't
                # fully zero out the relation pathway (see the warning in
                # __init__ -- this is what fixed the HIT@5 collapse observed
                # with real-timestamp decay on a sparse category).
                decay = floor + (1.0 - floor) * raw_decay
                decay = decay.unsqueeze(1)  # (B, 1, L, L, num_rel)
            else:
                # Fallback: sequence-position distance as a time proxy, shared
                # across the batch (used automatically when real timestamps
                # aren't available for this run -- see has_real_timestamps above).
                positions = torch.arange(seq_len, dtype=torch.float, device=input_tensor.device)
                pos_gap = (positions.unsqueeze(1) - positions.unsqueeze(0)).abs()  # (L, L)
                raw_decay = torch.exp(-decay_rate.view(1, 1, -1) * pos_gap.unsqueeze(-1))  # (L, L, num_rel)
                decay = floor + (1.0 - floor) * raw_decay
                decay = decay.unsqueeze(0).unsqueeze(0)  # (1, 1, L, L, num_rel)
            combined_weights = dynamic_weights * decay  # (B, 1, L, L, num_rel)
        else:
            combined_weights = dynamic_weights  # broadcasts over key-position dim

        rel_sum_relationship_att_scores = torch.sum(relationship_att_scores * combined_weights, dim=-1)  # (B, h, L, L)

        attention_probs = nn.Softmax(dim=-1)(attention_scores.clone() + rel_sum_relationship_att_scores)
        self.record_attention_diagnostics(rel_sum_relationship_att_scores, attention_probs)

        attention_probs = self.attn_dropout(attention_probs)
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        hidden_states = self.dense(context_layer)
        hidden_states = self.out_dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)

        return hidden_states


class DynamicRelationAwareLayer(nn.Module):
    def __init__(self, args, num_relationships):
        super(DynamicRelationAwareLayer, self).__init__()
        self.attention = DynamicRelationAwareSelfAttention(args, num_relationships)
        self.intermediate = Intermediate(args)

    def forward(self, hidden_states, attention_mask, input_relationships_masks, input_times=None):
        attention_output = self.attention(hidden_states, attention_mask, input_relationships_masks, input_times=input_times)
        intermediate_output = self.intermediate(attention_output)
        return intermediate_output


class DynamicRelationAwareSAEncoder(nn.Module):
    def __init__(self, args, num_relationships):
        super(DynamicRelationAwareSAEncoder, self).__init__()
        layer = DynamicRelationAwareLayer(args, num_relationships)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, input_relationships_masks, input_times=None, output_all_encoded_layers=True):
        all_encoder_layers = []
        relation_embs_all_layers = []
        relation_gates_all_layers = []

        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask, input_relationships_masks, input_times=input_times)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
                relation_embs_all_layers.append(layer_module.attention.relationship_embedding)
                # The "dynamic" analogue of the original's global relationship_weights:
                # here it's the gating linear layer itself (per-instance weights are
                # only materialized at forward time, so we expose the module that
                # produces them rather than a single fixed tensor).
                relation_gates_all_layers.append(layer_module.attention.relation_gate)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
            relation_embs_all_layers.append(self.layer[-1].attention.relationship_embedding)
            relation_gates_all_layers.append(self.layer[-1].attention.relation_gate)

        return all_encoder_layers, relation_embs_all_layers, relation_gates_all_layers


class Encoder(nn.Module):
    def __init__(self, args):
        super(Encoder, self).__init__()
        layer = Layer(args)
        self.layer = nn.ModuleList([copy.deepcopy(layer)
                                    for _ in range(args.num_hidden_layers)])

    def forward(self, hidden_states, attention_mask, output_all_encoded_layers=True):
        all_encoder_layers = []
        for layer_module in self.layer:
            hidden_states = layer_module(hidden_states, attention_mask)
            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)
        return all_encoder_layers
