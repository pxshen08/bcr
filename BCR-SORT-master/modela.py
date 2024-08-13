from antiberty import AntiBERTyRunner
import argparse
import torch
import numpy as np
import time
from Bio import SeqIO
import pandas as pd
import math
import torch.nn as nn
from torch.nn.utils.rnn import pad_sequence


class BCRSORT(nn.Module):
    def __init__(
            self,
            aa_embedding_dim,
            feature_embedding_dim,
            conv_out_channel,
            kernel_size,
            kernel_stride,
            conv_dilation,
            fc_dropout,
            hidden_fc1,
            hidden_fc2,
            antiBERTY_model_path=None
    ):
        super(BCRSORT, self).__init__()

        num_vgene = 55
        num_jgene = 7
        num_isotype = 6
        num_class = 3

        # Load pre-trained antiBERTy model
        self.antiBERTy = AntiBERTyRunner()

        # Define embeddings for vgene, jgene, and isotype
        self.emb_vgene = nn.Embedding(num_embeddings=num_vgene, embedding_dim=feature_embedding_dim)
        self.emb_jgene = nn.Embedding(num_embeddings=num_jgene, embedding_dim=feature_embedding_dim)
        self.emb_isotype = nn.Embedding(num_embeddings=num_isotype, embedding_dim=feature_embedding_dim)

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=aa_embedding_dim, out_channels=conv_out_channel,
                      kernel_size=kernel_size[0], stride=kernel_stride, dilation=conv_dilation),
            nn.BatchNorm1d(conv_out_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=aa_embedding_dim, out_channels=conv_out_channel,
                      kernel_size=kernel_size[1], stride=kernel_stride, dilation=conv_dilation),
            nn.BatchNorm1d(conv_out_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=aa_embedding_dim, out_channels=conv_out_channel,
                      kernel_size=kernel_size[2], stride=kernel_stride, dilation=conv_dilation),
            nn.BatchNorm1d(conv_out_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        input_dim = conv_out_channel * len(kernel_size) + feature_embedding_dim * 3
        nn1_out_dim, nn2_out_dim = hidden_fc1, hidden_fc2

        self.fc2 = nn.Sequential(
            nn.Linear(input_dim, nn1_out_dim),
            nn.BatchNorm1d(nn1_out_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout),
            nn.Linear(nn1_out_dim, nn2_out_dim),
            nn.BatchNorm1d(nn2_out_dim),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.fc_final = nn.Linear(nn2_out_dim, num_class)
        self.fc_auxiliary = nn.Sequential(
            nn.Linear(nn2_out_dim, 22),
            nn.BatchNorm1d(22),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

    def forward(self, sequence, vgene, jgene, isotype, batch_lengths, return_aux_out=False):
        # Process sequence through antiBERTy model
        # with torch.no_grad():
        encoded_sequence = self.antiBERTy.embed(sequence)  # Ensure sequence is in the format accepted by antiBERTy
        outputs_lstm_conv = torch.stack([a.mean(axis=0) for a in encoded_sequence])#(1024,512)
        # encoded_sequence1 = encoded_sequence.unsqueeze(0)
        # encoded_sequence1 =encoded_sequence1 .permute(0, 2, 1)
        #
        # lstm_conv1 = self.conv1d_1(encoded_sequence1)#(1,64,1)
        # lstm_conv1 = torch.squeeze(lstm_conv1, dim=2)#(1.64)
        #
        # lstm_conv2 = self.conv1d_2(encoded_sequence1)#(1,64,1)
        # lstm_conv2 = torch.squeeze(lstm_conv2, dim=2)#(1.64)
        #
        # lstm_conv3 = self.conv1d_3(encoded_sequence1)#(1,64,1)
        # lstm_conv3 = torch.squeeze(lstm_conv3, dim=2)#(1.64)
        # outputs_lstm_conv = torch.cat([lstm_conv1, lstm_conv2, lstm_conv3], dim=1)#(1,192)

        # Annotation features
        annotation = (vgene, jgene, isotype)
        outputs_afeature = self._annotation_forward(annotation)#(1024,192)
        expanded_outputs_lstm_conv = outputs_lstm_conv.expand(outputs_afeature.shape[0], -1)#(1024,192) gaihou:(1024,512)
        # Prediction
        x_fusion = torch.cat([expanded_outputs_lstm_conv, outputs_afeature], dim=1)#(1024,384)  gaihou:(1024,704)
        outputs_penultimate = self.fc2(x_fusion.float())#(1024,32)
        outputs = self.fc_final(outputs_penultimate)#(1024,3)
        outputs = nn.Softmax(dim=1)(outputs)#(1024,3)

        # Auxiliary loss
        if return_aux_out:
            outputs_aux = self._auxiliary_forward(outputs_penultimate)#(1024,22)
        else:
            outputs_aux = None

        return outputs, outputs_aux, outputs_penultimate, outputs_lstm_conv

    # def _annotation_forward(self, annotation_feature):
    #     vgene, jgene, isotype = annotation_feature
    #     vgene_emb = self.emb_vgene(vgene)
    #     jgene_emb = self.emb_jgene(jgene)
    #     isotype_emb = self.emb_isotype(isotype)
    #
    #     outputs = torch.cat([vgene_emb.squeeze(1), jgene_emb.squeeze(1), isotype_emb.squeeze(1)], dim=1)
    #     return outputs

    def _auxiliary_forward(self, x):
        output = self.fc_auxiliary(x)
        return nn.Softmax(dim=1)(output)

    def _annotation_forward(self, annotation_feature):
        vgene, jgene, isotype = annotation_feature

        # 确保输入张量在合理范围内
        if not all((vgene >= 0) & (vgene < self.emb_vgene.num_embeddings)):
            raise ValueError(f"vgene index out of range. vgene: {vgene}")

        if not all((jgene >= 0) & (jgene < self.emb_jgene.num_embeddings)):
            raise ValueError(f"jgene index out of range. jgene: {jgene}")

        if not all((isotype >= 0) & (isotype < self.emb_isotype.num_embeddings)):
            raise ValueError(f"isotype index out of range. isotype: {isotype}")
        vgene_emb = self.emb_vgene(vgene)
        jgene_emb = self.emb_jgene(jgene)
        isotype_emb = self.emb_isotype(isotype)

        outputs = torch.cat([vgene_emb.squeeze(1), jgene_emb.squeeze(1), isotype_emb.squeeze(1)], dim=1)

        return outputs


def load_model(args=None):
    if args is None:
        model = BCRSORT(aa_embedding_dim=512,  # Assuming antiBERTy embedding dimension
                        feature_embedding_dim=64,
                        conv_out_channel=64,
                        kernel_size=[3, 4, 5],
                        kernel_stride=1,
                        conv_dilation=1,
                        fc_dropout=0.5,
                        hidden_fc1=128,
                        hidden_fc2=32)
    else:
        model = BCRSORT(aa_embedding_dim=args.seq_dim, feature_embedding_dim=args.feature_dim,
                        conv_out_channel=args.num_channel, kernel_size=args.kernel_size,
                        kernel_stride=args.stride, conv_dilation=args.dilation,
                        fc_dropout=args.dropout_fc, hidden_fc1=args.dim_fc1, hidden_fc2=args.dim_fc2)

    return model
