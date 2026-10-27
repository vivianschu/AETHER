"""Architecture of the trained class-I binder in MHCAttnNet_dataset/models/classI/check.pt.

This is *not* MHCAttnNet (MHCAttnNet_ft/src/model.py): the checkpoint carries no
LSTM and no attention weights. Its 64 tensors are a Siamese transformer --

    peptide_embedding (23,100)        mhc_embedding (365,100)
    {pep,mhc}_pos_encoder.pos_encoding (500,1,100)
    {pep,mhc}_transformer_encoder.layers.{0,1}   d_model=100, ff=512
    peptide_linear0 (2048,4500)       mhc_linear0 (2048,12000)   <- dead weights
    peptide_linear (64,100)           mhc_linear (64,100)
    hidden_linear (64,128)            out_linear (2,64)

matching MHCAttnNet_ft/src/transformer.py's PositionalEncoding(max_len=500) and
TransformerEncoderLayer(dim_feedforward=512) with num_layers=2, combined with
the Linear(100, 64) arms of mlp.py's Transformer. The `*_linear0` layers belong
to the flattened route that is commented out in mlp.py; they are reproduced only
so the state dict loads strictly.

Two faithfully reproduced quirks of the training code -- see pepmhc/README.md:
  * PositionalEncoding slices `pos_encoding[:token_embedding.size(0)]` while the
    tensor is batch-first, so the offset is indexed by position *in the batch*.
  * nn.TransformerEncoder is left at batch_first=False and fed a batch-first
    tensor, so self-attention runs across the batch dimension.
Both make a score depend on the batch it was scored in. Predictor pins the
batch layout the model was trained under; see Predictor(batch_size=...).
"""
import math

import torch
import torch.nn as nn

PEPTIDE_LENGTH = 45          # config.PEPTIDE_LENGTH, in 1-grams
MHC_LENGTH = 120             # config.MHC_AMINO_ACID_LENGTH, in 3-grams
EMBED_DIM = 100
LINEAR0_OUT = 2048
LINEAR1_OUT = 64
LINEAR2_OUT = 2
POS_MAX_LEN = 500
DIM_FEEDFORWARD = 512
NUM_LAYERS = 2


class PositionalEncoding(nn.Module):
    """Verbatim from MHCAttnNet_ft/src/transformer.py, including the batch-first
    slicing quirk in forward()."""

    def __init__(self, dim_model, dropout_p, max_len):
        super().__init__()
        self.dropout = nn.Dropout(dropout_p)
        pos_encoding = torch.zeros(max_len, dim_model)
        positions_list = torch.arange(0, max_len, dtype=torch.float).view(-1, 1)
        division_term = torch.exp(
            torch.arange(0, dim_model, 2).float() * (-math.log(10000.0)) / dim_model)
        pos_encoding[:, 0::2] = torch.sin(positions_list * division_term)
        pos_encoding[:, 1::2] = torch.cos(positions_list * division_term)
        pos_encoding = pos_encoding.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pos_encoding", pos_encoding)

    def forward(self, token_embedding):
        return self.dropout(
            token_embedding + self.pos_encoding[:token_embedding.size(0), :])


class PepMHCTransformer(nn.Module):
    def __init__(self, pep_vocab=23, mhc_vocab=365, nhead=10,
                 dim_feedforward=DIM_FEEDFORWARD, num_layers=NUM_LAYERS,
                 dropout=0.2, activation="mish", seq_first=False,
                 relu_on_output=False):
        super().__init__()
        self.d_model = EMBED_DIM

        self.peptide_embedding = nn.Embedding(pep_vocab, EMBED_DIM)
        self.mhc_embedding = nn.Embedding(mhc_vocab, EMBED_DIM)

        self.peptide_linear0 = nn.Linear(PEPTIDE_LENGTH * EMBED_DIM, LINEAR0_OUT)
        self.mhc_linear0 = nn.Linear(MHC_LENGTH * EMBED_DIM, LINEAR0_OUT)

        self.peptide_linear = nn.Linear(EMBED_DIM, LINEAR1_OUT)
        self.mhc_linear = nn.Linear(EMBED_DIM, LINEAR1_OUT)
        self.hidden_linear = nn.Linear(LINEAR1_OUT * 2, LINEAR1_OUT)
        self.out_linear = nn.Linear(LINEAR1_OUT, LINEAR2_OUT)

        self.pep_pos_encoder = PositionalEncoding(EMBED_DIM, dropout, POS_MAX_LEN)
        self.mhc_pos_encoder = PositionalEncoding(EMBED_DIM, dropout, POS_MAX_LEN)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=EMBED_DIM, nhead=nhead,
            dim_feedforward=dim_feedforward, dropout=dropout)
        self.pep_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        self.mhc_transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)

        self.act = {"mish": nn.Mish(), "relu": nn.ReLU()}[activation]
        # seq_first=True feeds the encoder a (seq, batch, feature) tensor, i.e. the
        # wiring the author meant; False reproduces the batch-first tensor the
        # training code actually passed to a batch_first=False encoder.
        self.seq_first = seq_first
        # transformer.py ends relu(out_linear(hidden_linear(x))); mlp.py ends
        # out_linear(act(hidden_linear(x))). The checkpoint fits either shape.
        self.relu_on_output = relu_on_output
        self.relu = nn.ReLU()

    def _arm(self, ids, embedding, pos_encoder, encoder, linear):
        emb = embedding(ids) * math.sqrt(self.d_model)
        if self.seq_first:
            emb = emb.transpose(0, 1)
        emb = encoder(pos_encoder(emb))
        # pool over the sequence axis, wherever it currently is
        return self.act(linear(emb.mean(axis=0 if self.seq_first else 1)))

    def peptide_arm(self, peptide):
        return self._arm(peptide, self.peptide_embedding, self.pep_pos_encoder,
                         self.pep_transformer_encoder, self.peptide_linear)

    def mhc_arm(self, mhc):
        """Separated out so Predictor can cache it: for a batch of identical
        allele rows the result depends only on the allele and the batch size."""
        return self._arm(mhc, self.mhc_embedding, self.mhc_pos_encoder,
                         self.mhc_transformer_encoder, self.mhc_linear)

    def head(self, pep_out, mhc_out):
        conc = torch.cat((pep_out, mhc_out), dim=1)
        if self.relu_on_output:
            return self.relu(self.out_linear(self.hidden_linear(conc)))
        return self.out_linear(self.act(self.hidden_linear(conc)))

    def forward(self, peptide, mhc):
        return self.head(self.peptide_arm(peptide), self.mhc_arm(mhc))


# backwards-compatible alias
PepMHCNet = PepMHCTransformer
