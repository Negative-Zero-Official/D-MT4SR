import random
import numpy as np

import torch
from torch.utils.data import Dataset

from utils import neg_sample, neg_sample_popularity

class PretrainDataset(Dataset):

    def __init__(self, args, user_seq, long_sequence):
        self.args = args
        self.user_seq = user_seq
        self.long_sequence = long_sequence
        self.max_len = args.max_seq_length
        self.part_sequence = []
        self.split_sequence()

    def split_sequence(self):
        for seq in self.user_seq:
            input_ids = seq[-(self.max_len+2):-2] # keeping same as train set
            for i in range(len(input_ids)):
                self.part_sequence.append(input_ids[:i+1])

    def __len__(self):
        return len(self.part_sequence)

    def __getitem__(self, index):

        sequence = self.part_sequence[index] # pos_items
        # sample neg item for every masked item
        masked_item_sequence = []
        neg_items = []
        # Masked Item Prediction
        item_set = set(sequence)
        for item in sequence[:-1]:
            prob = random.random()
            if prob < self.args.mask_p:
                masked_item_sequence.append(self.args.mask_id)
                neg_items.append(neg_sample(item_set, self.args.item_size))
            else:
                masked_item_sequence.append(item)
                neg_items.append(item)

        # add mask at the last position
        masked_item_sequence.append(self.args.mask_id)
        neg_items.append(neg_sample(item_set, self.args.item_size))

        # Segment Prediction
        if len(sequence) < 2:
            masked_segment_sequence = sequence
            pos_segment = sequence
            neg_segment = sequence
        else:
            sample_length = random.randint(1, len(sequence) // 2)
            start_id = random.randint(0, len(sequence) - sample_length)
            neg_start_id = random.randint(0, len(self.long_sequence) - sample_length)
            pos_segment = sequence[start_id: start_id + sample_length]
            neg_segment = self.long_sequence[neg_start_id:neg_start_id + sample_length]
            masked_segment_sequence = sequence[:start_id] + [self.args.mask_id] * sample_length + sequence[
                                                                                      start_id + sample_length:]
            pos_segment = [self.args.mask_id] * start_id + pos_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))
            neg_segment = [self.args.mask_id] * start_id + neg_segment + [self.args.mask_id] * (
                        len(sequence) - (start_id + sample_length))

        assert len(masked_segment_sequence) == len(sequence)
        assert len(pos_segment) == len(sequence)
        assert len(neg_segment) == len(sequence)

        # padding sequence
        pad_len = self.max_len - len(sequence)
        masked_item_sequence = [0] * pad_len + masked_item_sequence
        pos_items = [0] * pad_len + sequence
        neg_items = [0] * pad_len + neg_items
        masked_segment_sequence = [0]*pad_len + masked_segment_sequence
        pos_segment = [0]*pad_len + pos_segment
        neg_segment = [0]*pad_len + neg_segment

        masked_item_sequence = masked_item_sequence[-self.max_len:]
        pos_items = pos_items[-self.max_len:]
        neg_items = neg_items[-self.max_len:]

        masked_segment_sequence = masked_segment_sequence[-self.max_len:]
        pos_segment = pos_segment[-self.max_len:]
        neg_segment = neg_segment[-self.max_len:]

        # Associated Attribute Prediction
        # Masked Attribute Prediction
        attributes = []
        for item in pos_items:
            attribute = [0] * self.args.attribute_size
            try:
                now_attribute = self.args.item2attribute[str(item)]
                for a in now_attribute:
                    attribute[a] = 1
            except:
                pass
            attributes.append(attribute)


        assert len(attributes) == self.max_len
        assert len(masked_item_sequence) == self.max_len
        assert len(pos_items) == self.max_len
        assert len(neg_items) == self.max_len
        assert len(masked_segment_sequence) == self.max_len
        assert len(pos_segment) == self.max_len
        assert len(neg_segment) == self.max_len


        cur_tensors = (torch.tensor(attributes, dtype=torch.long),
                       torch.tensor(masked_item_sequence, dtype=torch.long),
                       torch.tensor(pos_items, dtype=torch.long),
                       torch.tensor(neg_items, dtype=torch.long),
                       torch.tensor(masked_segment_sequence, dtype=torch.long),
                       torch.tensor(pos_segment, dtype=torch.long),
                       torch.tensor(neg_segment, dtype=torch.long),)
        return cur_tensors

class SASRecDataset(Dataset):

    def __init__(self, args, user_seq, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        if self.data_type == "train":
            input_ids = items[:-3]
            target_pos = items[1:-2]
            answer = [0] # no use

        elif self.data_type == 'valid':
            input_ids = items[:-2]
            target_pos = items[1:-1]
            answer = [items[-2]]

        else:
            input_ids = items[:-1]
            target_pos = items[1:]
            answer = [items[-1]]


        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(self.sample_negative(seq_set))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            )
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
            )

        return cur_tensors

    def sample_negative(self, seq_set):
        """Negative sampling hook. Default: uniform random, identical to original
        MT4SR/SASRec behavior. Overridden by D-MT4SR dataset variants for
        popularity-aware sampling."""
        return neg_sample(seq_set, self.args.item_size)

    def __len__(self):
        return len(self.user_seq)




class RelationAwareSASRecDataset(Dataset):

    def __init__(self, args, user_seq, relationship_mask_mat_fullseqs, relationships_ind_map, Item, test_neg_items=None, data_type='train'):
        self.args = args
        self.user_seq = user_seq
        self.relationship_mask_mat_fullseqs = relationship_mask_mat_fullseqs
        self.relationships_ind_map = relationships_ind_map
        self.Item = Item
        self.item_list_in_Item = list(self.Item.keys())
        self.test_neg_items = test_neg_items
        self.data_type = data_type
        self.max_len = args.max_seq_length

    def __getitem__(self, index):

        user_id = index
        items = self.user_seq[index]
        user_seq_mask_mat = self.relationship_mask_mat_fullseqs[index]

        _, acutual_seq_len, _ = user_seq_mask_mat.shape

        seq_mask = np.zeros((len(self.relationships_ind_map), self.max_len+1, self.max_len+1))

        item_rel_pos = np.zeros((self.max_len, len(self.relationships_ind_map)))
        item_rel = np.random.choice(self.item_list_in_Item, self.max_len, replace=False)
        for ind, eachitem in enumerate(item_rel):
            for rel_ind in range(len(self.relationships_ind_map)):
                if rel_ind in self.Item[eachitem]:
                    if len(self.Item[eachitem][rel_ind]) > 0:
                        item_rel_pos[ind, rel_ind] = np.random.choice(self.Item[eachitem][rel_ind], 1)[0]

        assert self.data_type in {"train", "valid", "test"}

        # [0, 1, 2, 3, 4, 5, 6]
        # train [0, 1, 2, 3]
        # target [1, 2, 3, 4]

        # valid [0, 1, 2, 3, 4]
        # answer [5]

        # test [0, 1, 2, 3, 4, 5]
        # answer [6]
        # NOTE: input_slice/target_slice are equivalent to the literal slicing
        # used before (items[:-3], items[1:-2], etc.) -- expressed as slice
        # objects so extra_tensors() (below) can apply the exact same window to
        # other per-position sequences (e.g. timestamps for D-MT4SR time decay).
        if self.data_type == "train":
            input_slice = slice(None, -3)
            target_slice = slice(1, -2)
            answer = [0] # no use
            if acutual_seq_len - 2 <= self.max_len+1:
                seq_mask[:, -(acutual_seq_len-2):, -(acutual_seq_len-2):] = user_seq_mask_mat[:, :-2, :-2]
            else:
                seq_mask[:, :, :] = user_seq_mask_mat[:, -(self.max_len+2+1):-2, -(self.max_len+2+1):-2]


        elif self.data_type == 'valid':
            input_slice = slice(None, -2)
            target_slice = slice(1, -1)
            answer = [items[-2]]
            if acutual_seq_len - 1 <= self.max_len+1:
                seq_mask[:, -(acutual_seq_len-1):, -(acutual_seq_len-1):] = user_seq_mask_mat[:, :-1, :-1]
            else:
                seq_mask[:, :, :] = user_seq_mask_mat[:, -(self.max_len+1+1):-1, -(self.max_len+1+1):-1]

        else:
            input_slice = slice(None, -1)
            target_slice = slice(1, None)
            answer = [items[-1]]
            if acutual_seq_len <= self.max_len+1:
                seq_mask[:, -acutual_seq_len:, -acutual_seq_len:] = user_seq_mask_mat[:, :, :]
            else:
                seq_mask[:, :, :] = user_seq_mask_mat[:, -(self.max_len+1):, -(self.max_len+1):]

        input_ids = items[input_slice]
        target_pos = items[target_slice]

        target_neg = []
        seq_set = set(items)
        for _ in input_ids:
            target_neg.append(self.sample_negative(seq_set))

        pad_len = self.max_len - len(input_ids)
        input_ids = [0] * pad_len + input_ids
        target_pos = [0] * pad_len + target_pos
        target_neg = [0] * pad_len + target_neg

        input_ids = input_ids[-self.max_len:]
        target_pos = target_pos[-self.max_len:]
        target_neg = target_neg[-self.max_len:]

        assert len(input_ids) == self.max_len
        assert len(target_pos) == self.max_len
        assert len(target_neg) == self.max_len

        # Hook for extra per-position tensors (e.g. D-MT4SR timestamps aligned
        # to input_ids via the same input_slice). Default: none, so the batch
        # structure below is byte-for-byte identical to the original MT4SR
        # dataset unless a subclass opts in.
        extra = self.extra_tensors(index, items, input_slice)

        if self.test_neg_items is not None:
            test_samples = self.test_neg_items[index]

            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long), # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(seq_mask, dtype=torch.long),
                torch.tensor(item_rel, dtype=torch.long),
                torch.tensor(item_rel_pos, dtype=torch.long),
                torch.tensor(test_samples, dtype=torch.long),
            ) + tuple(extra)
        else:
            cur_tensors = (
                torch.tensor(user_id, dtype=torch.long),  # user_id for testing
                torch.tensor(input_ids, dtype=torch.long),
                torch.tensor(target_pos, dtype=torch.long),
                torch.tensor(target_neg, dtype=torch.long),
                torch.tensor(answer, dtype=torch.long),
                torch.tensor(seq_mask, dtype=torch.long),
                torch.tensor(item_rel, dtype=torch.long),
                torch.tensor(item_rel_pos, dtype=torch.long),
            ) + tuple(extra)

        return cur_tensors

    def sample_negative(self, seq_set):
        """Negative sampling hook. Default: uniform random, identical to original
        MT4SR behavior. Overridden by DynamicRelationAwareSASRecDataset for
        popularity-aware sampling."""
        return neg_sample(seq_set, self.args.item_size)

    def extra_tensors(self, index, items, input_slice):
        """Hook for appending extra per-sample tensors to the batch tuple (e.g.
        timestamps for D-MT4SR time-decay attention). Default: no extra
        tensors, so RelationAwareSASRecDataset's batch structure is exactly the
        original MT4SR format. Overridden by DynamicRelationAwareSASRecDataset."""
        return []

    def __len__(self):
        return len(self.user_seq)


class DynamicRelationAwareSASRecDataset(RelationAwareSASRecDataset):
    """D-MT4SR dataset.

    Identical to RelationAwareSASRecDataset (same relation masks, same
    intra-/inter-sequence item_rel sampling) except it:

    1. Optionally swaps in popularity-aware negative sampling (see
       utils.neg_sample_popularity) when `args.popularity_neg_sampling` is set
       and a sampling distribution is provided.
    2. Optionally appends a per-position raw-timestamp tensor (aligned to
       input_ids) to the batch, when `user_seq_times` is available -- i.e. the
       loaded dataset was produced by the timestamp-aware preprocess_fromscratch.py.
       If `user_seq_times` is None (older preprocessed .npy file), an all-zero
       sentinel tensor is returned instead so the batch shape stays constant;
       DynamicRelationAwareSelfAttention only consults real timestamp values
       when args.has_real_timestamps is True, so this is a safe no-op fallback
       to the original position-distance decay proxy.

    Both additions are independently toggleable and isolated from the dynamic
    relation-weighting architecture change (which lives in the model), so each
    D-MT4SR contribution can be compared against the RelationAwareSASRecDataset
    baseline on its own.
    """

    def __init__(self, args, user_seq, relationship_mask_mat_fullseqs, relationships_ind_map, Item,
                 test_neg_items=None, data_type='train', sampling_probs=None, user_seq_times=None):
        super(DynamicRelationAwareSASRecDataset, self).__init__(
            args, user_seq, relationship_mask_mat_fullseqs, relationships_ind_map, Item,
            test_neg_items=test_neg_items, data_type=data_type)
        # Precomputed popularity sampling distribution (see utils.build_popularity_sampling_probs).
        # None => falls back to uniform sampling, identical to the original MT4SR dataset.
        self.sampling_probs = sampling_probs
        # Per-user raw interaction timestamps, index-aligned with user_seq (see
        # utils.get_user_seqs_MoHRdata). None if the loaded .npy predates
        # timestamp support -- extra_tensors() below degrades gracefully.
        self.user_seq_times = user_seq_times

    def sample_negative(self, seq_set):
        if getattr(self.args, 'popularity_neg_sampling', False) and self.sampling_probs is not None:
            # args.popneg_mix is the probability that a given negative is drawn
            # from the popularity distribution; the rest are uniform. 1.0 (the
            # default) is the original all-popularity behavior.
            #
            # Pure popularity sampling repeatedly pushes down the scores of
            # exactly the items that full-sort HIT/NDCG/MRR reward ranking
            # highly, which is a coherent explanation for why it hurt and why
            # those runs early-stopped so quickly. A mixture keeps the hard
            # negatives while leaving enough uniform mass that popular items
            # aren't systematically suppressed.
            mix = float(getattr(self.args, 'popneg_mix', 1.0))
            if mix >= 1.0 or random.random() < mix:
                return neg_sample_popularity(seq_set, self.args.item_size, self.sampling_probs)
            return neg_sample(seq_set, self.args.item_size)
        return super(DynamicRelationAwareSASRecDataset, self).sample_negative(seq_set)

    def extra_tensors(self, index, items, input_slice):
        if self.user_seq_times is not None:
            times = self.user_seq_times[index][input_slice]
            pad_len = self.max_len - len(times)
            times = [0.0] * pad_len + list(times)
            times = times[-self.max_len:]
        else:
            # No real timestamps available for this run (older preprocessed
            # file, or timestamps not requested) -- return an all-zero
            # sentinel so the batch structure is identical either way. The
            # model falls back to position-distance decay in this case.
            times = [0.0] * self.max_len
        return [torch.tensor(times, dtype=torch.float)]
