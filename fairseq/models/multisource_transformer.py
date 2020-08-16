from fairseq.models import register_model, register_model_architecture
from fairseq.models import MultisourceEncoder, SrcFactorEncoder
from fairseq.models.transformer import TransformerModel, TransformerEncoder

import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Any, Dict, List, Optional, Tuple

@register_model("multisource_transformer")
class MultisourceTransformerModel(TransformerModel):
    @classmethod
    def build_model(cls, args, task):
        base_architecture(args)

        args.encoder_embed_dim = args.encoder_src_dim

        if args.encoder_layers_to_keep:
            args.encoder_layers = len(args.encoder_layers_to_keep.split(","))
        if args.decoder_layers_to_keep:
            args.decoder_layers = len(args.decoder_layers_to_keep.split(","))

        if getattr(args, "max_source_positions", None) is None:
            args.max_source_positions = DEFAULT_MAX_SOURCE_POSITIONS
        if getattr(args, "max_target_positions", None) is None:
            args.max_target_positions = DEFAULT_MAX_TARGET_POSITIONS

        src_dict, tgt_dict = task.source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            if args.encoder_dim != args.decoder_embed_dim:
                raise ValueError(
                    "--share-all-embeddings requires --encoder-embed-dim to match --decoder-embed-dim"
                )
            if args.decoder_embed_path and (
                args.decoder_embed_path != args.encoder_embed_path
            ):
                raise ValueError(
                    "--share-all-embeddings not compatible with --decoder-embed-path"
                )
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = cls.build_embedding(
                args, src_dict, args.encoder_embed_dim, args.encoder_embed_path
            )
            decoder_embed_tokens = cls.build_embedding(
                args, tgt_dict, args.decoder_embed_dim, args.decoder_embed_path
            )

        # allow for src_factor
        src_factor_tokens = None
        if getattr(args,"encoder_src_factor_dim",False):
            src_factor_tokens = cls.build_embedding(args, src_dict, args.encoder_src_factor_dim)
            args.encoder_embed_dim = args.encoder_src_dim + args.encoder_src_factor_dim

        encoder = cls.build_encoder(args, src_dict, encoder_embed_tokens, src_factor_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(args, encoder, decoder)

    @staticmethod
    def add_args(parser):
        TransformerModel.add_args(parser)
        parser.add_argument('--share-encoder', action='store_true',
                            help='src and mt use the same encoder')

    @classmethod
    def build_encoder(cls, args, src_dict, embed_tokens, src_factor_embed_tokens = None):
        if src_factor_embed_tokens:
            src_encoder = SrcFactorEncoder(args, src_dict, embed_tokens,src_factor_embed_tokens)
        else:
            src_encoder = TransformerEncoder(args, src_dict, embed_tokens)
        if getattr(args, "apply_bert_init", False):
            src_encoder.apply(init_bert_params)

        if getattr(args, "share_encoder", False):
            mt_encoder = src_encoder
        else:
            if src_factor_embed_tokens:
                mt_encoder = SrcFactorEncoder(args, src_dict, embed_tokens,src_factor_embed_tokens)
            else:
                mt_encoder = TransformerEncoder(args, src_dict, embed_tokens)
            if getattr(args, "apply_bert_init", False):
                mt_encoder.apply(init_bert_params)
        encoder = MultisourceEncoder(src_encoder,{'mt':mt_encoder},order=['mt'])
        return encoder

    def forward(
        self,
        src_tokens,
        src_lengths,
        additional_tokens,
        additional_lengths,
        prev_output_tokens,
        src_factor_tokens = None,
        src_factor_lengths = None,
        additional_factor_tokens = None,
        additional_factor_lengths = None,
        return_all_hiddens: bool = True,
        features_only: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        encoder_out = self.encoder(
            src_tokens,
            src_lengths=src_lengths,
            additional_tokens=additional_tokens,
            additional_lengths=additional_lengths,
            src_factor_tokens=src_factor_tokens,
            src_factor_lengths=src_factor_lengths,
            additional_factor_tokens=additional_factor_tokens,
            additional_factor_lengths=additional_factor_lengths,
            return_all_hiddens=return_all_hiddens
        )
        decoder_out = self.decoder(
            prev_output_tokens,
            encoder_out=encoder_out,
            features_only=features_only,
            alignment_layer=alignment_layer,
            alignment_heads=alignment_heads,
            src_lengths=src_lengths,
            return_all_hiddens=return_all_hiddens,
        )
        return decoder_out

@register_model_architecture("multisource_transformer", "multisource_transformer")
def base_architecture(args):
    args.encoder_embed_path = getattr(args, "encoder_embed_path", None)
    # src_factor stuff
    args.encoder_src_dim = getattr(args, "encoder_src_dim", 512)
    args.encoder_src_factor_dim = getattr(args, "encoder_src_factor_dim",0)
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
    args.no_cross_attention = getattr(args, "no_cross_attention", False)
    args.cross_self_attention = getattr(args, "cross_self_attention", False)

    args.decoder_output_dim = getattr(
        args, "decoder_output_dim", args.decoder_embed_dim
    )
    args.decoder_input_dim = getattr(args, "decoder_input_dim", args.decoder_embed_dim)

    args.no_scale_embedding = getattr(args, "no_scale_embedding", False)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.tie_adaptive_weights = getattr(args, "tie_adaptive_weights", False)

@register_model_architecture("multisource_transformer", "multisource_transformer_src_factor")
def multisource_transformer_src_factor(args):
    args.encoder_src_dim = getattr(args, "encoder_src_dim", 512)
    args.encoder_src_factor_dim = getattr(args, "encoder_src_factor_dim",16)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 512)
    base_architecture(args)
