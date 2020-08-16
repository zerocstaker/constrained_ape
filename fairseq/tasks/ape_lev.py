import torch

from fairseq.tasks import register_task
from fairseq.tasks.translation_lev import TranslationLevenshteinTask
from fairseq import utils
from fairseq.tasks.ape import load_ape_dataset


@register_task('ape_lev')
class APELevenshteinTask(TranslationLevenshteinTask):

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        TranslationLevenshteinTask.add_args(parser)

        parser.add_argument(
            '--input-type',
            default='src-only',
            choices=['src-only', 'concatenate', 'multisource']
        )

        parser.add_argument(
            '--prev-target',
            default='target',
            choices=['target', 'mt', 'term']
        )

        parser.add_argument(
            '--init-output',
            default='blank',
            choices=['blank','mt','term']
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
        )

    def build_generator(self, models, args):
        # add models input to match the API for SequenceGenerator
        from fairseq.iterative_refinement_generator import IterativeRefinementGenerator
        init_tokens = None if args.init_output == 'blank' else args.init_output
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

        sample['prev_target'] = self.inject_noise(sample[self.args.prev_target])

        loss, sample_size, logging_output = criterion(model, sample)
        if ignore_grad:
            loss *= 0
        optimizer.backward(loss)
        return loss, sample_size, logging_output

    def valid_step(self, sample, model, criterion):
        model.eval()
        with torch.no_grad():
            sample['prev_target'] = self.inject_noise(sample[self.args.prev_target])

            loss, sample_size, logging_output = criterion(model, sample)
        return loss, sample_size, logging_output
