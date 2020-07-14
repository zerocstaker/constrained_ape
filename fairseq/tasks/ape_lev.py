import itertools
import logging
import os

import torch

from fairseq.utils import new_arange
from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq import utils
from fairseq.data import (
    AppendTokenDataset,
    ConcatDataset,
    data_utils,
    encoders,
    indexed_dataset,
    PrependTokenDataset,
    StripTokenDataset,
    TruncateDataset,
    APEDataset,
)

logger = logging.getLogger(__name__)

def load_ape_dataset(
    data_path, split,
    src_dict, tgt_dict,
    combine, dataset_impl, upsample_primary,
    left_pad_source, left_pad_target, max_source_positions,
    max_target_positions, prepend_bos=False, load_alignments=False,
    truncate_source=False, append_source_id=False,
    num_buckets=0,
    src_type="src",
):
    """
    ignoring src and tgt name. Assume $split.src, $split.mt, and $split.pe exist
    """
    src = src_type
    mt = "mt"
    tgt = "pe"
    term = "term"
    src_factor = src_type + "_embed"

    def split_exists(split, lang, data_path):
        filename = os.path.join(data_path, '{}.{}'.format(split, lang))
        return indexed_dataset.dataset_exists(filename, impl=dataset_impl)

    def load_dataset(lang, lang_dict, prefix, dataset_length, sample_ratios=None):
        """
        Function to load additional dataset and deal with all parameters.
        Easier than copying redudant code for each dataset.
        Requires src_dataset to provide the length and sample_ratios.
        """
        lang_datasets = []
        lang_dataset = data_utils.load_indexed_dataset(prefix + lang, lang_dict, dataset_impl)
        if lang_dataset is not None:
            lang_datasets.append(lang_dataset)
        assert dataset_length == len(lang_datasets) or len(lang_datasets) == 0
        if dataset_length == 1:
            lang_dataset = lang_datasets[0] if len(lang_datasets) > 0 else None
        else:
            assert sample_ratios is not None
            if len(lang_datasets) > 0:
                lang_dataset = ConcatDataset(lang_datasets, sample_ratios)
            else:
                lang_dataset = None
        if prepend_bos:
            assert hasattr(src_dict, "bos_index") and hasattr(lang_dict, "bos_index")
            if lang_dataset is not None:
                lang_dataset = PrependTokenDataset(lang_dataset, lang_dict.bos())
        eos = None
        if append_source_id:
            if lang_dataset is not None:
                lang_dataset = AppendTokenDataset(lang_dataset, lang_dict.index('[{}]'.format(lang)))

        lang_dataset_sizes = lang_dataset.sizes if lang_dataset is not None else None
        return lang_dataset, lang_dataset_sizes

    src_datasets = []
    for k in itertools.count():
        split_k = split + (str(k) if k > 0 else '')

        # infer langcode
        if split_exists(split_k, src, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))
        elif split_exists(split_k, mt, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))
        elif split_exists(split_k, tgt, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))
        elif split_exists(split_k, term, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))
        elif split_exists(split_k, src_factor, data_path):
            prefix = os.path.join(data_path, '{}.'.format(split_k))
        else:
            if k > 0:
                break
            else:
                raise FileNotFoundError('Dataset not found: {} ({})'.format(split, data_path))

        src_dataset = data_utils.load_indexed_dataset(prefix + src, src_dict, dataset_impl)
        if truncate_source:
            src_dataset = AppendTokenDataset(
                TruncateDataset(
                    StripTokenDataset(src_dataset, src_dict.eos()),
                    max_source_positions - 1,
                ),
                src_dict.eos(),
            )
        src_datasets.append(src_dataset)

        if not combine:
            break

    dataset_length = len(src_datasets)
    sample_ratios = None
    if len(src_datasets) == 1:
        src_dataset = src_datasets[0]
    else:
        sample_ratios = [1] * len(src_datasets)
        sample_ratios[0] = upsample_primary
        src_dataset = ConcatDataset(src_datasets, sample_ratios)

    if prepend_bos:
        assert hasattr(src_dict, "bos_index") and hasattr(tgt_dict, "bos_index")
        src_dataset = PrependTokenDataset(src_dataset, src_dict.bos())

    eos = None
    if append_source_id:
        src_dataset = AppendTokenDataset(src_dataset, src_dict.index('[{}]'.format(src)))
        eos = tgt_dict.index('[{}]'.format(tgt))

    align_dataset = None

    mt_dataset, mt_dataset_sizes = load_dataset(mt, tgt_dict, prefix, dataset_length, sample_ratios=sample_ratios)
    tgt_dataset, tgt_dataset_sizes = load_dataset(tgt, tgt_dict, prefix, dataset_length, sample_ratios=sample_ratios)
    term_dataset, term_dataset_sizes = load_dataset(term, tgt_dict, prefix, dataset_length, sample_ratios=sample_ratios)
    src_factor_dataset, src_factor_dataset_sizes = load_dataset(src_factor, tgt_dict, prefix, dataset_length, sample_ratios=sample_ratios)

    logger.info('{} {} {} examples'.format(
        data_path, split_k, len(src_datasets[-1])
    ))

    return APEDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        mt_dataset, mt_dataset_sizes,
        term_dataset, term_dataset_sizes,
        src_factor_dataset, src_factor_dataset_sizes,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
    )


@register_task('ape_lev')
class APELevenshteinTask(TranslationLevenshteinTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationLevenshteinTask.add_args(parser)
        parser.add_argument('--mt-as-output', action='store_true',
                            help='initialize output with mt')

        parser.add_argument('--mt-as-prev-target', action='store_true',
                            help='change prev_target to mt')

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        # infer langcode
        #src, tgt = self.args.source_lang, self.args.target_lang

        self.datasets[split] = load_ape_dataset(
            data_path, split,
            self.src_dict, self.tgt_dict,
            combine=combine, dataset_impl=self.args.dataset_impl,
            upsample_primary=self.args.upsample_primary,
            left_pad_source=self.args.left_pad_source,
            left_pad_target=self.args.left_pad_target,
            max_source_positions=self.args.max_source_positions,
            max_target_positions=self.args.max_target_positions,
            prepend_bos=True,
        )

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        init_tokens=None
        if getattr(args, 'mt_as_output', False):
            init_tokens = "mt"
        return IterativeRefinementGenerator(
            self.target_dictionary,
            eos_penalty=getattr(args, 'iter_decode_eos_penalty', 0.0),
            max_iter=getattr(args, 'iter_decode_max_iter', 10),
            beam_size=getattr(args, 'iter_decode_with_beam', 1),
            reranking=getattr(args, 'iter_decode_with_external_reranker', False),
            decoding_format=getattr(args, 'decoding_format', None),
            adaptive=not getattr(args, 'iter_decode_force_max_iter', False),
            retain_history=getattr(args, 'retain_iter_history', False),
            init_tokens=init_tokens
        )

    def train_step(self,
                   sample,
                   model,
                   criterion,
                   optimizer,
                   update_num,
                   ignore_grad=False):
        model.train()

        if self.args.mt_as_prev_target:
            sample['prev_target'] = self.inject_noise(sample['mt'])
        else:
            sample['prev_target'] = self.inject_noise(sample['target'])

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            if self.args.mt_as_prev_target:
                sample['prev_target'] = self.inject_noise(sample['mt'])
            else:
                sample['prev_target'] = self.inject_noise(sample['target'])

            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
