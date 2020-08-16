
from fairseq.models.nat.levenshtein_transformer import LevenshteinTransformerModel
from fairseq.models import register_model, register_model_architecture
from fairseq.models import (
    FairseqEncoder,
)
from fairseq.models.nat import (
    FairseqNATEncoder
)
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.models import MultisourceEncoder


import torch
import torch.nn as nn
import torch.nn.functional as F

from .levenshtein_utils import (
    _skip, _skip_encoder_out, _fill,
    _get_ins_targets, _get_del_targets,
    _apply_ins_masks, _apply_ins_words, _apply_del_words
)

@register_model("multisource_levenshtein_transformer")
class MultisourceLevenshteinTransformerModel(LevenshteinTransformerModel):

    @staticmethod
    def add_args(parser):
        LevenshteinTransformerModel.add_args(parser)
        parser.add_argument('--share-encoder', action='store_true',
                            help='src and mt use the same encoder')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens):
        src_encoder = FairseqNATEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            src_encoder.apply(init_bert_params)

        if getattr(args, "share_encoder", False):
            mt_encoder = src_encoder
        else:
            mt_encoder = FairseqNATEncoder(args, src_dict, embed_tokens)

        encoder = MultisourceEncoder(src_encoder,{'mt':mt_encoder},order=['mt'])
        return encoder

    def forward(
        self, src_tokens, src_lengths, additional_tokens, additional_lengths, prev_output_tokens, tgt_tokens, **kwargs
    ):

        assert tgt_tokens is not None, "forward function only supports training."

        # fill pad when initiaze with mt that is longer than pe
        if prev_output_tokens.size(1) > tgt_tokens.size(1):
            pads = tgt_tokens.new_full((tgt_tokens.size(0),prev_output_tokens.size(1) - tgt_tokens.size(1)),self.pad)
            tgt_tokens = torch.cat([tgt_tokens,pads],1)

        # encoding
        encoder_out = self.encoder(src_tokens, src_lengths=src_lengths, additional_tokens=additional_tokens, additional_lengths=additional_lengths, **kwargs)

        # generate training labels for insertion
        masked_tgt_masks, masked_tgt_tokens, mask_ins_targets = _get_ins_targets(
            prev_output_tokens, tgt_tokens, self.pad, self.unk
        )
        mask_ins_targets = mask_ins_targets.clamp(min=0, max=255)  # for safe prediction
        mask_ins_masks = prev_output_tokens[:, 1:].ne(self.pad)

        mask_ins_out, _ = self.decoder.forward_mask_ins(
            normalize=False,
            prev_output_tokens=prev_output_tokens,
            encoder_out=encoder_out
        )
        word_ins_out, _ = self.decoder.forward_word_ins(
            normalize=False,
            prev_output_tokens=masked_tgt_tokens,
            encoder_out=encoder_out
        )

        # make online prediction
        if self.decoder.sampling_for_deletion:
            word_predictions = torch.multinomial(
                F.softmax(word_ins_out, -1).view(-1, word_ins_out.size(-1)), 1).view(
                    word_ins_out.size(0), -1)
        else:
            word_predictions = F.log_softmax(word_ins_out, dim=-1).max(2)[1]

        word_predictions.masked_scatter_(
            ~masked_tgt_masks, tgt_tokens[~masked_tgt_masks]
        )

        # generate training labels for deletion
        word_del_targets = _get_del_targets(word_predictions, tgt_tokens, self.pad)
        word_del_out, _ = self.decoder.forward_word_del(
            normalize=False,
            prev_output_tokens=word_predictions,
            encoder_out=encoder_out)
        word_del_masks = word_predictions.ne(self.pad)

        return {
            "mask_ins": {
                "out": mask_ins_out, "tgt": mask_ins_targets,
                "mask": mask_ins_masks, "ls": 0.01,
            },
            "word_ins": {
                "out": word_ins_out, "tgt": tgt_tokens,
                "mask": masked_tgt_masks, "ls": self.args.label_smoothing,
                "nll_loss": True
            },
            "word_del": {
                "out": word_del_out, "tgt": word_del_targets,
                "mask": word_del_masks
            }
        }

@register_model_architecture("multisource_levenshtein_transformer", "multisource_levenshtein_transformer")
def levenshtein_base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 2048)
    args.encoder_layers = getattr(args, "encoder_layers", 6)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 8)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", False)
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", False)
    args.decoder_embed_path = getattr(args, "decoder_embed_path", None)
    args.decoder_embed_dim = getattr(args, "decoder_embed_dim", args.encoder_embed_dim)
    args.decoder_ffn_embed_dim = getattr(
        args, "decoder_ffn_embed_dim", args.encoder_ffn_embed_dim
    )
    args.decoder_layers = getattr(args, "decoder_layers", 6)
    args.decoder_attention_heads = getattr(args, "decoder_attention_heads", 8)
    args.decoder_normalize_before = getattr(args, "decoder_normalize_before", False)
    args.decoder_learned_pos = getattr(args, "decoder_learned_pos", False)
    args.attention_dropout = getattr(args, "attention_dropout", 0.0)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.activation_fn = getattr(args, "activation_fn", "relu")
    args.dropout = getattr(args, "dropout", 0.1)
    args.adaptive_softmax_cutoff = getattr(args, "adaptive_softmax_cutoff", None)
    args.adaptive_softmax_dropout = getattr(args, "adaptive_softmax_dropout", 0)
    args.share_decoder_input_output_embed = getattr(
        args, "share_decoder_input_output_embed", False
    )
    args.share_all_embeddings = getattr(args, "share_all_embeddings", False)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )
    args.adaptive_input = getattr(args, "adaptive_input", False)
    args.apply_bert_init = getattr(args, "apply_bert_init", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.sampling_for_deletion = getattr(args, "sampling_for_deletion", False)
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)
    args.early_exit = getattr(args, "early_exit", "6,6,6")
    args.no_share_discriminator = getattr(args, "no_share_discriminator", False)
    args.no_share_maskpredictor = getattr(args, "no_share_maskpredictor", False)
    args.share_discriminator_maskpredictor = getattr(args, "share_discriminator_maskpredictor", False)
    args.no_share_last_layer = getattr(args, "no_share_last_layer", False)

    args.share_encoder = getattr(args, "share_encoder", False)
