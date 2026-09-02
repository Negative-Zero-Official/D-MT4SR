# -*- coding: utf-8 -*-

import argparse
import numpy as np
import math
import random
import os
import json
import pickle
from collections import deque
from scipy.sparse import csr_matrix
from tqdm import tqdm
import multiprocessing

import torch
import torch.nn.functional as F


class ModelEMA:
    """Exponential moving average of model weights.

    Maintains a shadow copy of every floating-point parameter, updated after
    each optimizer step as  shadow = decay * shadow + (1 - decay) * param.
    Integer buffers and non-float parameters are copied verbatim.

    Used for model selection rather than for training: `copy_to()` swaps the
    averaged weights into the live model for evaluation and `restore()` puts
    the raw weights back, so training itself is completely unchanged. This is
    a pure variance-reduction device -- an averaged model is far less
    sensitive to which particular minibatch the last update happened to see,
    which is exactly the noise that makes single-epoch validation scores (and
    therefore checkpoint selection) unstable on small datasets.
    """

    def __init__(self, model, decay=0.999):
        self.decay = float(decay)
        self.shadow = {k: v.detach().clone()
                       for k, v in model.state_dict().items()}
        self._backup = None

    @torch.no_grad()
    def update(self, model):
        for name, value in model.state_dict().items():
            shadow = self.shadow.get(name)
            if shadow is None or not shadow.is_floating_point():
                self.shadow[name] = value.detach().clone()
                continue
            shadow.mul_(self.decay).add_(value.detach(), alpha=1.0 - self.decay)

    def state_dict(self):
        return self.shadow

    @torch.no_grad()
    def copy_to(self, model):
        """Swaps the averaged weights into `model`, saving the live ones."""
        self._backup = {k: v.detach().clone()
                        for k, v in model.state_dict().items()}
        model.load_state_dict(self.shadow)

    @torch.no_grad()
    def restore(self, model):
        """Undoes copy_to(). No-op if copy_to() wasn't called."""
        if self._backup is None:
            return
        model.load_state_dict(self._backup)
        self._backup = None

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # some cudnn methods can be random even after fixing the seed
    # unless you tell it to be deterministic
    torch.backends.cudnn.deterministic = True

def check_path(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print(f'{path} created')

def neg_sample(item_set, item_size):
    item = random.randint(1, item_size - 1)
    while item in item_set:
        item = random.randint(1, item_size - 1)
    return item

class EarlyStopping:
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, checkpoint_path, patience=7, verbose=False, delta=0,
                 smooth_window=1):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
            smooth_window (int): Number of recent epochs to average the monitored
                            score over before comparing. 1 = original behavior
                            (compare the raw per-epoch score).

                            On small datasets per-epoch validation MRR is noisy
                            enough that argmax-over-epochs partly selects lucky
                            epochs rather than good models -- and a
                            higher-variance model gets more lucky epochs, which
                            inflates both its reported score and its seed
                            spread. Averaging over a short window makes model
                            selection track the underlying trend instead.
        """
        self.checkpoint_path = checkpoint_path
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.delta = delta
        self.smooth_window = max(1, int(smooth_window))
        self._history = deque(maxlen=self.smooth_window)

    def _smooth(self, score):
        """Averages the last `smooth_window` scores element-wise."""
        if self.smooth_window == 1:
            return score
        self._history.append(np.asarray(score, dtype=float))
        return np.mean(np.stack(list(self._history), axis=0), axis=0)

    def compare(self, score):
        for i in range(len(score)):
            if score[i] > self.best_score[i]+self.delta:
                return False
        return True

    def __call__(self, score, model):
        # score HIT@10 NDCG@10
        score = self._smooth(score)

        if self.best_score is None:
            self.best_score = score
            self.score_min = np.array([0]*len(score))
            self.save_checkpoint(score, model)
        elif self.compare(score):
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(score, model)
            self.counter = 0

    def save_checkpoint(self, score, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print(f'Validation score increased.  Saving model ...')
        torch.save(model.state_dict(), self.checkpoint_path)
        self.score_min = score

def kmax_pooling(x, dim, k):
    index = x.topk(k, dim=dim)[1].sort(dim=dim)[0]
    return x.gather(dim, index).squeeze(dim)

def avg_pooling(x, dim):
    return x.sum(dim=dim)/x.size(dim)


def generate_rating_matrix_valid(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-2]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def generate_rating_matrix_test(user_seq, num_users, num_items):
    # three lists are used to construct sparse matrix
    row = []
    col = []
    data = []
    for user_id, item_list in enumerate(user_seq):
        for item in item_list[:-1]: #
            row.append(user_id)
            col.append(item)
            data.append(1)

    row = np.array(row)
    col = np.array(col)
    data = np.array(data)
    rating_matrix = csr_matrix((data, (row, col)), shape=(num_users, num_items))

    return rating_matrix

def get_user_seqs(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    num_users = len(lines)
    num_items = max_item + 2

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users

def get_user_seqs_MoHRdata(dataset):
    dataset = np.load('./data/'+dataset+'Partitioned_5core.npy', allow_pickle=True)

    # D-MT4SR: preprocess_fromscratch.py now appends three extra timestamp dicts
    # (user_train_times, user_validation_times, user_testing_times) after the
    # original 8 fields. Detect which format is loaded so this function keeps
    # working unchanged on .npy files generated before this change.
    has_times = len(dataset) >= 11
    if has_times:
        [user_train, user_validation, user_test, Item, Item_relationship_mask_mat_completeseqs, relationships_ind, usernum, itemnum,
         user_train_times, user_validation_times, user_testing_times] = dataset[:11]
    else:
        [user_train, user_validation, user_test, Item, Item_relationship_mask_mat_completeseqs, relationships_ind, usernum, itemnum] = dataset[:8]
        user_train_times = user_validation_times = user_testing_times = None

    user_seq = []
    user_seq_mask_mat_rel = []
    # user_seq_times mirrors user_seq exactly (same per-user length, same order)
    # when timestamps are available; None otherwise, signalling downstream code
    # (DynamicRelationAwareSASRecDataset / DynamicRelationAwareSelfAttention) to
    # fall back to its position-distance decay proxy.
    user_seq_times = [] if has_times else None
    for uid in user_train.keys():
        seq = [itemid+1 for itemid in user_train[uid]]
        if has_times:
            seq_times = list(user_train_times[uid])
        if len(user_validation) == 2:
            seq.append(user_validation[uid][1]+1)
            if has_times:
                seq_times.append(user_validation_times[uid][1])
        else:
            seq.append(user_validation[uid][0]+1)
            if has_times:
                seq_times.append(user_validation_times[uid][0])
        if len(user_test) == 2:
            seq.append(user_test[uid][1]+1)
            if has_times:
                seq_times.append(user_testing_times[uid][1])
        else:
            seq.append(user_test[uid][0]+1)
            if has_times:
                seq_times.append(user_testing_times[uid][0])
        user_seq.append(seq)
        user_seq_mask_mat_rel.append(Item_relationship_mask_mat_completeseqs[uid])
        if has_times:
            user_seq_times.append(seq_times)
    num_users = usernum
    max_item = itemnum - 1
    num_items = max_item + 2

    new_Item = {}
    for item, related_dict in Item.items():
        reindex_item = item + 1
        new_related_dict = {}
        for rel, rel_i_list in related_dict['related'].items():
            rel_ind = relationships_ind[rel]
            reindex_rel_i_list = [itemind+1 for itemind in rel_i_list]
            new_related_dict[rel_ind] = reindex_rel_i_list
        new_Item[reindex_item] = new_related_dict

    valid_rating_matrix = generate_rating_matrix_valid(user_seq, num_users, num_items)
    test_rating_matrix = generate_rating_matrix_test(user_seq, num_users, num_items)
    return user_seq, max_item, valid_rating_matrix, test_rating_matrix, num_users, user_seq_mask_mat_rel, relationships_ind, new_Item, user_seq_times

def get_user_seqs_long(data_file):
    lines = open(data_file).readlines()
    user_seq = []
    long_sequence = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        long_sequence.extend(items) # 
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    return user_seq, max_item, long_sequence

def get_user_seqs_and_sample(data_file, sample_file):
    lines = open(data_file).readlines()
    user_seq = []
    item_set = set()
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        user_seq.append(items)
        item_set = item_set | set(items)
    max_item = max(item_set)

    lines = open(sample_file).readlines()
    sample_seq = []
    for line in lines:
        user, items = line.strip().split(' ', 1)
        items = items.split(' ')
        items = [int(item) for item in items]
        sample_seq.append(items)

    assert len(user_seq) == len(sample_seq)

    return user_seq, max_item, sample_seq

def get_item2attribute_json(data_file):
    item2attribute = json.loads(open(data_file).readline())
    attribute_set = set()
    for item, attributes in item2attribute.items():
        attribute_set = attribute_set | set(attributes)
    attribute_size = max(attribute_set) # 331
    return item2attribute, attribute_size

def get_metric(pred_list, topk=10):
    NDCG = 0.0
    HIT = 0.0
    MRR = 0.0
    # [batch] the answer's rank
    for rank in pred_list:
        MRR += 1.0 / (rank + 1.0)
        if rank < topk:
            NDCG += 1.0 / np.log2(rank + 2.0)
            HIT += 1.0
    return HIT /len(pred_list), NDCG /len(pred_list), MRR /len(pred_list)

def precision_at_k_per_sample(actual, predicted, topk):
    num_hits = 0
    for place in predicted:
        if place in actual:
            num_hits += 1
    return num_hits / (topk + 0.0)

def precision_at_k(actual, predicted, topk):
    sum_precision = 0.0
    num_users = len(predicted)
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        sum_precision += len(act_set & pred_set) / float(topk)

    return sum_precision / num_users

def recall_at_k(actual, predicted, topk):
    sum_recall = 0.0
    num_users = len(predicted)
    true_users = 0
    recall_dict = {}
    for i in range(num_users):
        act_set = set(actual[i])
        pred_set = set(predicted[i][:topk])
        if len(act_set) != 0:
            #sum_recall += len(act_set & pred_set) / float(len(act_set))
            one_user_recall = len(act_set & pred_set) / float(len(act_set))
            recall_dict[i] = one_user_recall
            sum_recall += one_user_recall
            true_users += 1
    return sum_recall / true_users, recall_dict

def cal_mrr(actual, predicted):
    sum_mrr = 0.
    true_users = 0
    num_users = len(predicted)
    mrr_dict = {}
    for i in range(num_users):
        r = []
        act_set = set(actual[i])
        pred_list = predicted[i]
        for item in pred_list:
            if item in act_set:
                r.append(1)
            else:
                r.append(0)
        r = np.array(r)
        if np.sum(r) > 0:
            #sum_mrr += np.reciprocal(np.where(r==1)[0]+1, dtype=np.float)[0]
            one_user_mrr = np.reciprocal(np.where(r==1)[0]+1, dtype=float)[0]
            sum_mrr += one_user_mrr
            true_users += 1
            mrr_dict[i] = one_user_mrr
        else:
            mrr_dict[i] = 0.
    return sum_mrr / len(predicted), mrr_dict


def apk(actual, predicted, k=10):
    """
    Computes the average precision at k.
    This function computes the average precision at k between two lists of
    items.
    Parameters
    ----------
    actual : list
             A list of elements that are to be predicted (order doesn't matter)
    predicted : list
                A list of predicted elements (order does matter)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The average precision at k over the input lists
    """
    if len(predicted)>k:
        predicted = predicted[:k]

    score = 0.0
    num_hits = 0.0

    for i,p in enumerate(predicted):
        if p in actual and p not in predicted[:i]:
            num_hits += 1.0
            score += num_hits / (i+1.0)

    if not actual:
        return 0.0

    return score / min(len(actual), k)


def mapk(actual, predicted, k=10):
    """
    Computes the mean average precision at k.
    This function computes the mean average prescision at k between two lists
    of lists of items.
    Parameters
    ----------
    actual : list
             A list of lists of elements that are to be predicted
             (order doesn't matter in the lists)
    predicted : list
                A list of lists of predicted elements
                (order matters in the lists)
    k : int, optional
        The maximum number of predicted elements
    Returns
    -------
    score : double
            The mean average precision at k over the input lists
    """
    return np.mean([apk(a, p, k) for a, p in zip(actual, predicted)])

def ndcg_k(actual, predicted, topk):
    res = 0
    ndcg_dict = {}
    for user_id in range(len(actual)):
        k = min(topk, len(actual[user_id]))
        idcg = idcg_k(k)
        dcg_k = sum([int(predicted[user_id][j] in
                         set(actual[user_id])) / math.log(j+2, 2) for j in range(topk)])
        res += dcg_k / idcg
        ndcg_dict[user_id] = dcg_k / idcg
    return res / float(len(actual)), ndcg_dict


# Calculates the ideal discounted cumulative gain at k
def idcg_k(k):
    res = sum([1.0/math.log(i+2, 2) for i in range(k)])
    if not res:
        return 1.0
    else:
        return res

def dcg_at_k(r, k, method=1):
    """Score is discounted cumulative gain (dcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Discounted cumulative gain
    """
    r = np.asfarray(r)[:k]
    if r.size:
        if method == 0:
            return r[0] + np.sum(r[1:] / np.log2(np.arange(2, r.size + 1)))
        elif method == 1:
            return np.sum(r / np.log2(np.arange(2, r.size + 2)))
        else:
            raise ValueError('method must be 0 or 1.')
    return 0.


def ndcg_at_k(r, k, method=1):
    """Score is normalized discounted cumulative gain (ndcg)
    Relevance is positive real values.  Can use binary
    as the previous methods.
    Returns:
        Normalized discounted cumulative gain
    """
    dcg_max = dcg_at_k(sorted(r, reverse=True), k, method)
    if not dcg_max:
        return 0.
    return dcg_at_k(r, k, method) / dcg_max


def itemperf_recall(ranks, k):
    ranks = np.array(ranks)
    if len(ranks) == 0:
        return 0
    return np.sum(ranks<=k) / len(ranks)

def itemperf_ndcg(ranks, k, size):
    ndcg = 0.0
    if len(ranks) == 0:
        return 0.
    for onerank in ranks:
        r = np.zeros(size)
        r[onerank-1] = 1
        ndcg += ndcg_at_k(r, k)
    return ndcg / len(ranks)


def get_user_performance_perpopularity(train, results_users, Ks):
    [recall_dict_list, ndcg_dict_list, mrr_dict] = results_users
    short_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short_seqs = 0

    long_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_long_seqs = 0

    short7_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short7_seqs = 0

    short37_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_short37_seqs = 0

    medium3_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }

    num_medium3_seqs = 0

    medium7_seq_results = {
            "recall": np.zeros(len(Ks)),
            "ndcg": np.zeros(len(Ks)),
            "mrr": 0.,
    }
    num_medium7_seqs = 0

    test_users = list(train.keys())
    for result_user in tqdm(test_users):
        if len(train[result_user]) <= 3:
            num_short_seqs += 1
        if len(train[result_user]) <= 7:
            num_short7_seqs += 1
        if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
            num_short37_seqs += 1
        if len(train[result_user]) > 3 and len(train[result_user]) < 20:
            num_medium3_seqs += 1
        if len(train[result_user]) > 7 and len(train[result_user]) < 20:
            num_medium7_seqs += 1
        if len(train[result_user]) >= 20:
            num_long_seqs += 1
    for k_ind in range(len(recall_dict_list)):
        k = Ks[k_ind]
        recall_dict_k = recall_dict_list[k_ind]
        ndcg_dict_k = ndcg_dict_list[k_ind]

        for result_user in tqdm(test_users):
            if len(train[result_user]) <= 3:
                short_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) <= 7:
                short7_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short7_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
                short37_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                short37_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 3 and len(train[result_user]) < 20:
                medium3_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                medium3_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) > 7 and len(train[result_user]) < 20:
                medium7_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                medium7_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

            if len(train[result_user]) >= 20:
                long_seq_results["recall"][k_ind] += recall_dict_k[result_user]
                long_seq_results["ndcg"][k_ind] += ndcg_dict_k[result_user]

    for result_user in tqdm(test_users):
        if len(train[result_user]) <= 3:
            short_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) <= 7:
            short7_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 3 and len(train[result_user]) <= 7:
            short37_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 3 and len(train[result_user]) < 20:
            medium3_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) > 7 and len(train[result_user]) < 20:
            medium7_seq_results["mrr"] += mrr_dict[result_user]

        if len(train[result_user]) >= 20:
            long_seq_results["mrr"] += mrr_dict[result_user]

    if num_short_seqs > 0:
        short_seq_results["recall"] /= num_short_seqs
        short_seq_results["ndcg"] /= num_short_seqs
        short_seq_results["mrr"] /= num_short_seqs
    print(f"testing #of short seq users with less than 3 training points: {num_short_seqs}")

    if num_short7_seqs > 0:
        short7_seq_results["recall"] /= num_short7_seqs
        short7_seq_results["ndcg"] /= num_short7_seqs
        short7_seq_results["mrr"] /= num_short7_seqs
    print(f"testing #of short seq users with less than 7 training points: {num_short7_seqs}")

    if num_short37_seqs > 0:
        short37_seq_results["recall"] /= num_short37_seqs
        short37_seq_results["ndcg"] /= num_short37_seqs
        short37_seq_results["mrr"] /= num_short37_seqs
    print(f"testing #of short seq users with 3 - 7 training points: {num_short37_seqs}")

    if num_medium3_seqs > 0:
        medium3_seq_results["recall"] /= num_medium3_seqs
        medium3_seq_results["ndcg"] /= num_medium3_seqs
        medium3_seq_results["mrr"] /= num_medium3_seqs
    print(f"testing #of short seq users with medium3 training points: {num_medium3_seqs}")

    if num_medium7_seqs > 0:
        medium7_seq_results["recall"] /= num_medium7_seqs
        medium7_seq_results["ndcg"] /= num_medium7_seqs
        medium7_seq_results["mrr"] /= num_medium7_seqs
    print(f"testing #of short seq users with medium7 training points: {num_medium7_seqs}")

    if num_long_seqs > 0:
        long_seq_results["recall"] /= num_long_seqs
        long_seq_results["ndcg"] /= num_long_seqs
        long_seq_results["mrr"] /= num_long_seqs

    print(f"testing #of short seq users with >= 20 training points: {num_long_seqs}")

    print('testshort: ' + str(short_seq_results))
    print('testshort7: ' + str(short7_seq_results))
    print('testshort37: ' + str(short37_seq_results))
    print('testmedium3: ' + str(medium3_seq_results))
    print('testmedium7: ' + str(medium7_seq_results))
    print('testlong: ' + str(long_seq_results))


def eval_one_setitems(x):
    Ks = [1, 5, 10, 15, 20, 40]
    result = {
            "recall": 0,
            "ndcg": 0
    }
    ranks = x[0]
    k_ind = x[1]
    test_num_items = x[2]
    freq_ind = x[3]

    result['recall'] = itemperf_recall(ranks, Ks[k_ind])
    result['ndcg'] = itemperf_ndcg(ranks, Ks[k_ind], test_num_items)

    return result, k_ind, freq_ind


def get_item_performance_perpopularity(items_in_freqintervals, all_pos_items_ranks, Ks, freq_quantiles, num_items):
    cores = multiprocessing.cpu_count() // 2
    pool = multiprocessing.Pool(cores)
    test_num_items_in_intervals = []
    interval_results = [{'recall': np.zeros(len(Ks)), 'ndcg': np.zeros(len(Ks))} for _ in range(len(items_in_freqintervals))]

    all_freq_all_ranks = []
    all_ks = []
    all_numtestitems = []
    all_freq_ind = []
    for freq_ind, item_list in enumerate(items_in_freqintervals):
        num_item_pos_interactions = 0
        all_ranks = []
        interval_items = []
        for item in item_list:
            pos_ranks_oneitem = all_pos_items_ranks.get(item, [])
            if len(pos_ranks_oneitem) > 0:
                interval_items.append(item)
            all_ranks.extend(pos_ranks_oneitem)
        for k_ind in range(len(Ks)):
            all_ks.append(k_ind)
            all_freq_all_ranks.append(all_ranks)
            all_numtestitems.append(num_items)
            all_freq_ind.append(freq_ind)
        test_num_items_in_intervals.append(interval_items)

    item_eval_freq_data = zip(all_freq_all_ranks, all_ks, all_numtestitems, all_freq_ind)
    batch_item_result = pool.map(eval_one_setitems, item_eval_freq_data)


    for oneresult in batch_item_result:
        result_dict = oneresult[0]
        k_ind = oneresult[1]
        freq_ind = oneresult[2]
        interval_results[freq_ind]['recall'][k_ind] = result_dict['recall']
        interval_results[freq_ind]['ndcg'][k_ind] = result_dict['ndcg']



    item_freq = freq_quantiles
    for i in range(len(item_freq)+1):
        if i == 0:
            print('For items in freq between 0 - %d with %d items: ' % (item_freq[i], len(test_num_items_in_intervals[i])))
        elif i == len(item_freq):
            print('For items in freq between %d - max with %d items: ' % (item_freq[i-1], len(test_num_items_in_intervals[i])))
        else:
            print('For items in freq between %d - %d with %d items: ' % (item_freq[i-1], item_freq[i], len(test_num_items_in_intervals[i])))
        for k_ind in range(len(Ks)):
            k = Ks[k_ind]
            print('Recall@%d:%.6f, NDCG@%d:%.6f'%(k, interval_results[i]['recall'][k_ind], k, interval_results[i]['ndcg'][k_ind]))


# ---------------------------------------------------------------------------
# D-MT4SR additions: popularity-aware negative sampling
#
# The original MT4SR/SASRec pipeline samples negatives uniformly at random via
# `neg_sample`. Uniform negatives are mostly "easy" (very unlikely to be
# confused with the positive item), which caps how informative the gradient
# signal is. D-MT4SR optionally replaces this with popularity-aware sampling
# (the standard word2vec-style freq^0.75 trick), which biases sampling towards
# popular items and produces harder, more informative negatives. This is
# purely additive: `neg_sample` is untouched, so the original MT4SR models are
# unaffected, and the new sampler is only used when a dataset/trainer opts in.
# ---------------------------------------------------------------------------

def compute_item_popularity(user_seq, max_item):
    """Computes raw interaction frequency for each item id in [0, max_item+1].

    Item id 0 is reserved for padding and is not a real item. This mirrors the
    item id space used throughout datasets.py/main.py (`args.item_size = max_item + 2`).
    """
    freq = np.zeros(max_item + 2)
    for seq in user_seq:
        for item in seq:
            freq[item] += 1
    return freq


def build_popularity_sampling_probs(item_freq, power=0.75):
    """Builds a smoothed sampling distribution over items for negative sampling.

    Uses the standard freq^power smoothing (power=0.75 by default, as in
    word2vec negative sampling) so that popular items are oversampled relative
    to uniform sampling, but rare items still retain some sampling mass.
    Item id 0 (padding) is always excluded. Returns None if there is no usable
    signal (e.g., all-zero frequencies), in which case callers should fall
    back to uniform `neg_sample`.
    """
    probs = np.power(item_freq, power)
    probs[0] = 0.0
    total = probs.sum()
    if total <= 0:
        return None
    return probs / total


def neg_sample_popularity(item_set, item_size, sampling_probs):
    """Popularity-aware negative sampling for D-MT4SR.

    Draws a candidate item from `sampling_probs` (see build_popularity_sampling_probs)
    and rejects it if it collides with an item already in `item_set` (i.e., it's
    actually a positive for this sequence). Falls back to uniform `neg_sample`
    if no sampling distribution is available or rejection sampling stalls,
    guaranteeing this never hangs or breaks the training loop.
    """
    if sampling_probs is None:
        return neg_sample(item_set, item_size)

    item = np.random.choice(len(sampling_probs), p=sampling_probs)
    tries = 0
    while item in item_set or item == 0:
        item = np.random.choice(len(sampling_probs), p=sampling_probs)
        tries += 1
        if tries > 50:
            return neg_sample(item_set, item_size)
    return item


# ---------------------------------------------------------------------------
# Automatic --rel_loss_chunk_size selection
#
# The inter-sequence relation loss materializes a (batch*seq*num_rel,
# item_size) logits tensor. For Office_Products (item_size ~136k) at
# batch=128, seq=100, num_rel=2 that is 25600 x 136077 float32 ~= 13.9 GB for
# the forward pass alone, and roughly 2-3x that once the backward pass holds
# its intermediates -- hence ~28-42 GB. --rel_loss_chunk_size bounds that peak
# by computing the same sum in row-chunks under gradient checkpointing.
#
# Chunking is MATHEMATICALLY NEUTRAL: identical loss value, identical
# gradients (see RelationAwareSASRecModelTrainer.relation_outside_seq_loss).
# The only cost is recomputation time in the backward pass, so the right
# policy is "use the largest chunk the card can hold, and none at all if it
# can hold everything".
#
# Thresholds are on TOTAL device memory, keeping the shipped numbers explicit
# rather than derived, so a run's chosen value can be reproduced from the log.
# ---------------------------------------------------------------------------
# MEASURED, not guessed. The chunk's peak is about THREE copies of the
# (chunk, item_size) tensor live at once -- cross_entropy holds the logits, the
# log_softmax it computes from them, and the incoming gradient -- so
#
#     peak_bytes ~= 3 * chunk * item_size * 4
#
# For Office_Products (item_size 136,077) that is 1.56 MiB per row. Preflight
# on an RTX 5070 measured 13.1 GiB peak at chunk 8192 against a predicted
# 12.5 GiB + model, confirming the factor of 3.
#
# The policy is to keep that under ~70% of total VRAM, leaving room for the
# model, activations and allocator fragmentation. An earlier version of this
# table was one tier too aggressive at every level (8192 on a 12 GiB card needs
# 12.5 GiB and silently spills into host memory on Windows, or OOMs on Linux).
CHUNK_SIZE_BY_VRAM_GB = (
    # (minimum total VRAM in GiB, chunk size; 0 = unchunked)
    (70.0, 0),       # 80 GB A100/H100: unchunked needs ~39 GiB at batch 128 -- fits
    (35.0, 16384),   # 40 GB A100 / 48 GB L40S  (~24.9 GiB)
    (20.0, 8192),    # 24 GB 3090 / 4090 / A5000 (~12.5 GiB)
    (0.0,  4096),    # 12-16 GB consumer cards, e.g. RTX 5070 (~6.2 GiB)
)

# Bytes of peak GPU memory per chunk row, per item in the catalog. Used to
# predict the peak before allocating it, so preflight can say whether the
# chosen chunk size actually fits rather than finding out by OOMing.
CHUNK_PEAK_BYTES_PER_ROW_PER_ITEM = 3 * 4


def predicted_chunk_peak_gib(chunk_size, item_size, batch_size=128,
                             max_seq_length=100, num_rel=2):
    """Predicted peak GiB of the inter-sequence relation loss.

    chunk_size 0 (unchunked) is scored at the full row count, which is what
    that setting actually materializes.
    """
    rows = batch_size * max_seq_length * num_rel
    effective = rows if not chunk_size or chunk_size <= 0 else min(chunk_size, rows)
    return (effective * item_size * CHUNK_PEAK_BYTES_PER_ROW_PER_ITEM) / (1024 ** 3)

# Sentinel stored in args.rel_loss_chunk_size by the '--rel_loss_chunk_size=auto'
# argparse type below, before the device is known. Resolved to a real int by
# resolve_rel_loss_chunk_size(). Negative so it can never be mistaken for a
# valid chunk size, and so the trainer's "<= 0 means unchunked" guard would
# fail safe rather than silently running unchunked if resolution were skipped.
REL_LOSS_CHUNK_AUTO = -1


def rel_loss_chunk_size_arg(value):
    """argparse type for --rel_loss_chunk_size: an int, or the string 'auto'.

    Returning an int in both cases keeps `str(args)` (which main.py writes as
    the first line of every log file) showing a plain integer, so logs written
    before and after this option existed stay directly comparable.
    """
    if isinstance(value, str) and value.strip().lower() == 'auto':
        return REL_LOSS_CHUNK_AUTO
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        raise argparse.ArgumentTypeError(
            f"--rel_loss_chunk_size expects an integer or 'auto', got {value!r}")
    if parsed < 0:
        raise argparse.ArgumentTypeError(
            "--rel_loss_chunk_size must be >= 0 (0 disables chunking), "
            f"got {parsed}")
    return parsed


def detect_total_vram_gb(device_index=0):
    """Total memory of the selected CUDA device in GiB, or None if no CUDA."""
    if not torch.cuda.is_available():
        return None
    try:
        props = torch.cuda.get_device_properties(device_index)
    except (AssertionError, RuntimeError):
        return None
    return props.total_memory / (1024 ** 3)


def chunk_size_for_vram(vram_gb):
    """Map total VRAM (GiB) to a chunk size using CHUNK_SIZE_BY_VRAM_GB."""
    for threshold, chunk in CHUNK_SIZE_BY_VRAM_GB:
        if vram_gb >= threshold:
            return chunk
    return CHUNK_SIZE_BY_VRAM_GB[-1][1]


def resolve_rel_loss_chunk_size(args, verbose=True):
    """Replace the 'auto' sentinel in args with a concrete chunk size.

    No-op for an explicitly given value, so a run that passes a number gets
    exactly that number and nothing about existing runs changes. Returns the
    resolved int.
    """
    requested = getattr(args, 'rel_loss_chunk_size', 0)
    if requested != REL_LOSS_CHUNK_AUTO:
        return requested

    vram_gb = detect_total_vram_gb()
    if vram_gb is None:
        # CPU-only run: there is no VRAM to bound, and chunking would only add
        # recomputation cost. Host RAM is the constraint instead, so pick the
        # smallest shipped chunk rather than unchunked.
        chunk = CHUNK_SIZE_BY_VRAM_GB[-1][1]
        if verbose:
            print(f"[rel_loss_chunk_size=auto] no CUDA device visible; "
                  f"using {chunk} to bound host RAM.", flush=True)
    else:
        chunk = chunk_size_for_vram(vram_gb)
        name = torch.cuda.get_device_name(0)
        how = 'unchunked (the full logits tensor fits)' if chunk == 0 else f'chunk size {chunk}'
        if verbose:
            print(f"[rel_loss_chunk_size=auto] detected {name} with "
                  f"{vram_gb:.1f} GiB VRAM -> {how}.", flush=True)

    args.rel_loss_chunk_size = chunk
    return chunk
