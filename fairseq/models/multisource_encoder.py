import torch

from fairseq.models import FairseqEncoder
from fairseq.models.fairseq_encoder import EncoderOut

class MultisourceEncoder(FairseqEncoder):
    """
    A wrapper around a dictionary of :class:`FairseqEncoder` objects.

    Very similiar to CompositeEncoder, but each encoder takes separate inputs
    and is then concatenated together. Assume a src_encoder to be the main encoder
    for easiness to use generate and other things that depends on src_tokens

    Args:
        encoders (dict): a dictionary of :class:`FairseqEncoder` objects.
        order (list): list of order to loop through the keys
    """

    def __init__(self, src_encoder, encoders, order):
        super().__init__(src_encoder.dictionary)
        assert "src" not in encoders
        self.encoders = {**{'src': src_encoder},**encoders}
        for key in self.encoders:
            self.add_module(key+"_encoder", self.encoders[key])
        for key in order:
            assert key in self.encoders
        self.order = order

    def forward(self, src_tokens, src_lengths, additional_tokens, additional_lengths, src_factor_tokens=None, src_factor_lengths=None, additional_factor_tokens=None, additional_factor_lengths=None, return_all_hiddens: bool = False):
        encoder_outs = []
        src_out = self.encoders['src'](src_tokens,src_lengths, src_factor_tokens, src_factor_lengths) if src_factor_tokens is not None else self.encoders['src'](src_tokens,src_lengths)
        encoder_outs.append(src_out)
        for key in self.order:
            assert key in additional_tokens and key in additional_lengths
            t = additional_tokens[key]
            l = additional_lengths[key]
            ft, fl = None, None
            if additional_factor_tokens is not None:
                ft = additional_factor_tokens[key]
                fl = additional_factor_lengths[key]
            add_out = self.encoders[key](t, l,ft,fl) if ft is not None else self.encoders[key](t, l)
            encoder_outs.append(add_out)
        return self.combine_encoder_out(encoder_outs)

    def combine_encoder_out(self, outs):
        encoder_out = torch.cat([out[0] for out in outs], 0)
        encoder_padding_mask = torch.cat([out[1] for out in outs], 1)
        encoder_embedding = torch.cat([out[2] for out in outs], 1)
        encoder_states = None
        if all( out[3] is not None for out in outs ):
            encoder_states = torch.cat([out[3] for out in outs], 0)
        return EncoderOut(
            encoder_out=encoder_out,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=None,
            src_lengths=None
        )

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        return min([encoder.max_positions() for encoder in self.encoders.values()])

    def upgrade_state_dict(self, state_dict):
        for encoder in self.encoders.values():
            encoder.upgrade_state_dict(state_dict)
        return state_dict

    @torch.jit.export
    def reorder_encoder_out(self, encoder_out: EncoderOut, new_order):
        """
        Taking from TransformerEncoder's reorder_out.

        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        """
        Since encoder_padding_mask and encoder_embedding are both of type
        Optional[Tensor] in EncoderOut, they need to be copied as local
        variables for Torchscript Optional refinement
        """
        encoder_padding_mask: Optional[Tensor] = encoder_out.encoder_padding_mask
        encoder_embedding: Optional[Tensor] = encoder_out.encoder_embedding

        new_encoder_out = (
            encoder_out.encoder_out
            if encoder_out.encoder_out is None
            else encoder_out.encoder_out.index_select(1, new_order)
        )
        new_encoder_padding_mask = (
            encoder_padding_mask
            if encoder_padding_mask is None
            else encoder_padding_mask.index_select(0, new_order)
        )
        new_encoder_embedding = (
            encoder_embedding
            if encoder_embedding is None
            else encoder_embedding.index_select(0, new_order)
        )
        src_tokens = encoder_out.src_tokens
        if src_tokens is not None:
            src_tokens = src_tokens.index_select(0, new_order)

        src_lengths = encoder_out.src_lengths
        if src_lengths is not None:
            src_lengths = src_lengths.index_select(0, new_order)

        encoder_states = encoder_out.encoder_states
        if encoder_states is not None:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return EncoderOut(
            encoder_out=new_encoder_out,  # T x B x C
            encoder_padding_mask=new_encoder_padding_mask,  # B x T
            encoder_embedding=new_encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
            src_tokens=src_tokens,  # B x T
            src_lengths=src_lengths,  # B x 1
        )
