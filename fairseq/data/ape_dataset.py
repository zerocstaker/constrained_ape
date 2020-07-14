import logging

import numpy as np
import torch

from fairseq.data import data_utils, LanguagePairDataset

logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=True,
    left_pad_target=False,
    input_feeding=True,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx, eos_idx, left_pad, move_eos_to_beginning,
        )

    id = torch.LongTensor([s['id'] for s in samples])
    src_tokens = merge('source', left_pad=left_pad_source)
    # sort by descending source length
    src_lengths = torch.LongTensor([
        s['source'].ne(pad_idx).long().sum() for s in samples
    ])
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    prev_output_tokens = None
    target = None
    if samples[0].get('target', None) is not None:
        target = merge('target', left_pad=left_pad_target)
        target = target.index_select(0, sort_order)
        tgt_lengths = torch.LongTensor([
            s['target'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)
        ntokens = tgt_lengths.sum().item()

        if input_feeding:
            # we create a shifted version of targets for feeding the
            # previous output token(s) into the next decoder step
            prev_output_tokens = merge(
                'target',
                left_pad=left_pad_target,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, sort_order)
    else:
        ntokens = src_lengths.sum().item()

    mt = None
    if samples[0].get('mt', None) is not None:
        mt = merge('mt', left_pad=left_pad_target)
        mt = mt.index_select(0, sort_order)
        mt_lengths = torch.LongTensor([
            s['mt'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)

    term = None
    if samples[0].get('term', None) is not None:
        term = merge('term', left_pad=left_pad_target)
        term = term.index_select(0, sort_order)
        term_lengths = torch.LongTensor([
            s['term'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)

    src_factor = None
    if samples[0].get('src_factor', None) is not None:
        src_factor = merge('src_factor', left_pad=left_pad_target)
        src_factor = src_factor.index_select(0, sort_order)
        src_factor_lengths = torch.LongTensor([
            s['src_factor'].ne(pad_idx).long().sum() for s in samples
        ]).index_select(0, sort_order)

    batch = {
        'id': id,
        'nsentences': len(samples),
        'ntokens': ntokens,
        'net_input': {
            'src_tokens': src_tokens,
            'src_lengths': src_lengths,
        },
        'target': target,
    }
    if prev_output_tokens is not None:
        batch['net_input']['prev_output_tokens'] = prev_output_tokens

    # ignoring alignment

    if mt is not None:
        batch['mt'] = mt
    if term is not None:
        batch['term'] = term
    if src_factor is not None:
        batch['net_input']['src_factor'] = src_factor

    return batch

class APEDataset(LanguagePairDataset):
    def __init__(
        self, src, src_sizes, src_dict,
        tgt=None, tgt_sizes=None, tgt_dict=None,
        mt=None, mt_sizes=None,
        term=None, term_sizes=None,
        src_factor=None, src_factor_sizes=None,
        left_pad_source=True, left_pad_target=False,
        shuffle=True, input_feeding=True,
        remove_eos_from_source=False, append_eos_to_target=False,
        align_dataset=None,
        append_bos=False, eos=None,
        num_buckets=0
    ):
        """
        Add mt to LanguagePairDataset

        Additional Args:
            mt (torch.utils.data.Dataset, optional): mt dataset to wrap
            mt_sizes (List[int], optional): mt sentence lengths
        """
        super().__init__(
            src, src_sizes, src_dict, tgt = tgt, tgt_sizes = tgt_sizes, tgt_dict = tgt_dict,
            left_pad_source = left_pad_source, left_pad_target = left_pad_target,
            shuffle = shuffle, input_feeding = input_feeding,
            remove_eos_from_source = remove_eos_from_source, append_eos_to_target = append_eos_to_target,
            align_dataset = align_dataset,
            append_bos = append_bos, eos = eos,
            num_buckets = num_buckets,
        )
        self.mt = mt
        self.mt_sizes = np.array(mt_sizes) if mt_sizes is not None else None
        self.term = term
        self.term_sizes = np.array(term_sizes) if term_sizes is not None else None
        self.src_factor = src_factor
        self.src_factor_sizes = np.array(src_factor_sizes) if src_factor_sizes is not None else None

        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset
            if self.mt is not None:
                self.mt = BucketPadLengthDataset(
                    self.mt,
                    sizes=self.mt_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.mt_sizes = self.mt.sizes
                logger.info('bucketing mt lengths: {}'.format(list(self.mt.buckets)))

            if self.term is not None:
                self.term = BucketPadLengthDataset(
                    self.term,
                    sizes=self.term_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.term_sizes = self.term.sizes
                logger.info('bucketing term lengths: {}'.format(list(self.term.buckets)))

            if self.src_factor is not None:
                self.src_factor = BucketPadLengthDataset(
                    self.src_factor,
                    sizes=self.src_factor_sizes,
                    num_buckets=num_buckets,
                    pad_idx=self.tgt_dict.pad(),
                    left_pad=self.left_pad_target,
                )
                self.src_factor_sizes = self.src_factor.sizes
                logger.info('bucketing src_factor lengths: {}'.format(list(self.src_factor.buckets)))

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        example = super().__getitem__(index)

        mt_item = self.mt[index] if self.mt is not None else None
        term_item = self.term[index] if self.term is not None else None
        src_factor_item = self.src_factor[index] if self.src_factor is not None else None

        # Append EOS to end of tgt sentence if it does not have an EOS and remove
        # EOS from end of src sentence if it exists. This is useful when we use
        # use existing datasets for opposite directions i.e., when we want to
        # use tgt_dataset as src_dataset and vice versa
        if self.append_eos_to_target:
            eos = self.tgt_dict.eos() if self.tgt_dict else self.src_dict.eos()
            if self.mt and self.mt[index][-1] != eos:
                mt_item = torch.cat([self.mt[index], torch.LongTensor([eos])])
            if self.term and self.term[index][-1] != eos:
                term_item = torch.cat([self.term[index], torch.LongTensor([eos])])
            if self.src_factor and self.src_factor[index][-1] != eos:
                src_factor_item = torch.cat([self.src_factor[index], torch.LongTensor([eos])])

        if self.append_bos:
            bos = self.tgt_dict.bos() if self.tgt_dict else self.src_dict.bos()
            if self.mt and self.mt[index][0] != bos:
                mt_item = torch.cat([torch.LongTensor([bos]), self.mt[index]])
            if self.term and self.term[index][0] != bos:
                term_item = torch.cat([torch.LongTensor([bos]), self.term[index]])
            if self.src_factor and self.src_factor[index][0] != bos:
                src_factor_item = torch.cat([torch.LongTensor([bos]), self.src_factor[index]])

        example["mt"] = mt_item
        example["term"] = term_item
        example["src_factor"] = src_factor_item

        return example

    def collater(self, samples):
        """Merge a list of samples to form a mini-batch."""
        return collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
        )

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return max(
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.mt_sizes[index] if self.mt_sizes is not None else 0,
            self.term_sizes[index] if self.term_sizes is not None else 0,
            self.src_factor_sizes[index] if self.src_factor_sizes is not None else 0
        )

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
            self.mt_sizes[index] if self.mt_sizes is not None else 0,
            self.term_sizes[index] if self.term_sizes is not None else 0,
            self.src_factor_sizes[index] if self.src_factor_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self))
        else:
            indices = np.arange(len(self))
        if self.buckets is None:
            # sort by target length, mt_length then source length
            if self.tgt_sizes is not None:
                indices = indices[
                    np.argsort(self.tgt_sizes[indices], kind='mergesort')
                ]
            if self.mt_sizes is not None:
                indices = indices[
                    np.argsort(self.mt_sizes[indices], kind='mergesort')
                ]
            if self.term_sizes is not None:
                indices = indices[
                    np.argsort(self.term_sizes[indices], kind='mergesort')
                ]
            if self.src_factor_sizes is not None:
                indices = indices[
                    np.argsort(self.src_factor_sizes[indices], kind='mergesort')
                ]
            return indices[np.argsort(self.src_sizes[indices], kind='mergesort')]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind='mergesort')
            ]

    @property
    def supports_prefetch(self):
        return (
            getattr(self.src, 'supports_prefetch', False)
            and (getattr(self.tgt, 'supports_prefetch', False) or self.tgt is None)
            and (getattr(self.mt, 'supports_prefetch', False) or self.mt is None)
            and (getattr(self.term, 'supports_prefetch', False) or self.mt is None)
            and (getattr(self.src_factor, 'supports_prefetch', False) or self.mt is None)
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.mt is not None:
            self.mt.prefetch(indices)
        if self.term is not None:
            self.term.prefetch(indices)
        if self.src_factor is not None:
            self.src_factor.prefetch(indices)
