import itertools
import logging
import os

import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationTask
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
    input_type='src_only',
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
    mt_factor = "mt_embed"

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
    mt_factor_dataset, mt_factor_dataset_sizes = load_dataset(mt_factor, tgt_dict, prefix, dataset_length, sample_ratios=sample_ratios)

    logger.info('{} {} {} examples'.format(
        data_path, split_k, len(src_datasets[-1])
    ))

    return APEDataset(
        src_dataset, src_dataset.sizes, src_dict,
        tgt_dataset, tgt_dataset_sizes, tgt_dict,
        mt_dataset, mt_dataset_sizes,
        term_dataset, term_dataset_sizes,
        src_factor_dataset, src_factor_dataset_sizes,
        mt_factor_dataset, mt_factor_dataset_sizes,
        left_pad_source=left_pad_source,
        left_pad_target=left_pad_target,
        align_dataset=align_dataset, eos=eos,
        num_buckets=num_buckets,
        input_type=input_type
    )

@register_task('ape')
class APETask(TranslationTask):
    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationTask.add_args(parser)

        parser.add_argument(
            '--input-type',
            default='src-only',
            choices=['src-only', 'concatenate', 'multisource']
        )

        parser.add_argument(
            '--src-type',
            default='src',
            choices=['src','src_append','src_replace']
        )

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
            input_type=self.args.input_type,
            src_type=self.args.src_type,
        )
