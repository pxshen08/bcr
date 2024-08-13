import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from torch.nn import LSTM
from antiberty import AntiBERTyRunner
from data_utils import amino_acid_dict

class BCRSORT(torch.nn.Module):
    def __init__(
            self,
            aa_embedding_dim,
            feature_embedding_dim,
            hidden_size,
            num_layers,
            lstm_dropout,
            proj_size,
            conv_out_channel,
            kernel_size,
            kernel_stride,
            conv_dilation,
            fc_dropout,
            hidden_fc1,
            hidden_fc2,
            antiberty
            ):
        super(BCRSORT, self).__init__()

        num_vgene = 60
        num_jgene = 7
        num_isotype = 6
        num_class = 3
        num_aa= 22

        self.antiBERTy = AntiBERTyRunner()
        self.AA_embedding = nn.Embedding(num_embeddings=num_aa, embedding_dim=aa_embedding_dim, padding_idx=0)
        self.emb_vgene = nn.Embedding(num_embeddings=num_vgene, embedding_dim=feature_embedding_dim)
        self.emb_jgene = nn.Embedding(num_embeddings=num_jgene, embedding_dim=feature_embedding_dim)
        self.emb_isotype = nn.Embedding(num_embeddings=num_isotype, embedding_dim=feature_embedding_dim)

        self.lstm = LSTM(input_size=aa_embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                         dropout=lstm_dropout, bidirectional=True, proj_size=proj_size, batch_first=True)

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_size, out_channels=conv_out_channel,
                      kernel_size=kernel_size[0], stride=kernel_stride, dilation=conv_dilation),
            nn.BatchNorm1d(conv_out_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_size, out_channels=conv_out_channel,
                      kernel_size=kernel_size[1], stride=kernel_stride, dilation=conv_dilation),
            nn.BatchNorm1d(conv_out_channel),
            nn.ReLU(),
            nn.AdaptiveMaxPool1d(output_size=1),
        )

        self.conv1d_3 = nn.Sequential(
            nn.Conv1d(in_channels=2 * hidden_size, out_channels=conv_out_channel,
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
            nn.Linear(nn2_out_dim, num_aa),
            nn.BatchNorm1d(num_aa),
            nn.ReLU(),
            nn.Dropout(fc_dropout)
        )

        self.global_avg_pool = nn.AdaptiveAvgPool1d(1)
        self.linear_reduce = nn.Linear(512, 256)

    def forward(self, sequence, vgene, jgene, isotype, batch_lengths, return_aux_out=False):
        # sequence
        encoded_sequence = self.antiBERTy.embed(sequence)  # Ensure sequence is in the format accepted by antiBERTy
        encoded_sequenceq = torch.stack([a.mean(axis=0) for a in encoded_sequence])  # (1024,512)
        outputs_lstm, outputs_lstm_length = self._lstm_forward(encoded_sequenceq, batch_lengths)
        #
        lstm_conv1 = self.conv1d_1(outputs_lstm.permute(0, 2, 1))
        lstm_conv1 = torch.squeeze(lstm_conv1, dim=2)

        lstm_conv2 = self.conv1d_2(outputs_lstm.permute(0, 2, 1))
        lstm_conv2 = torch.squeeze(lstm_conv2, dim=2)

        lstm_conv3 = self.conv1d_3(outputs_lstm.permute(0, 2, 1))
        lstm_conv3 = torch.squeeze(lstm_conv3, dim=2)
        outputs_lstm_conv = torch.cat([lstm_conv1, lstm_conv2, lstm_conv3], dim=1)#512.192

        # annotation features
        # annotation = (vgene, jgene, isotype)
        # outputs_afeature = self._annotation_forward(annotation)#(512,192)
        # outputs_lstm_conv=nn.Linear(encoded_sequenceq.shape[1], outputs_afeature.shape[1]).cuda()(encoded_sequenceq)#512.192
        # prediction
        # x_fusion = torch.cat([outputs_lstm_conv, outputs_afeature], dim=1)#512,384
        x_fusion = torch.cat([outputs_lstm_conv, outputs_lstm_conv], dim=1)  # 512,384
        outputs_penultimate = self.fc2(x_fusion.float())
        outputs = self.fc_final(outputs_penultimate)
        outputs = nn.Softmax(dim=1)(outputs)

        # auxiliary loss
        if return_aux_out is True:
            outputs_aux = self.antiBERTy.fill_masks(sequence)
            # Find the filled letter
            filled_letter = None
            filled_letters = []
            for orig, filled in zip(sequence, outputs_aux):
                for o, f in zip(orig, filled):
                    if o == '_':  # Assuming 'X' represents the missing letter to be filled
                        filled_letters.append(f)
                        break
            outputs_aux_one_hot = torch.zeros(len(filled_letters), len(amino_acid_dict)+1)
            for i, aa in enumerate(filled_letters):
                index = amino_acid_dict[aa]
                outputs_aux_one_hot[i, index - 1] = 1
            # outputs_aux = self._auxiliary_forward(outputs_penultimate)
        else:
            outputs_aux_one_hot = None

        return outputs, outputs_aux_one_hot, outputs_penultimate, outputs_lstm_conv

    def _lstm_forward(self, inputs, batch_lengths):
        # Use antiBERTy embeddings instead of AA_embedding
        # embedded_inputs1 = self.antiBERTy.embed(inputs)#(8232)
        # if isinstance(embedded_inputs1, list):
        #     embedded_inputs = torch.stack(embedded_inputs1)#(1024,3,512)
        # Apply global average pooling to convert [batch_size, seq_len, embed_dim] to [batch_size, embed_dim]
        # embedded_inputs = self.global_avg_pool(inputs.permute(0, 2, 1)).squeeze(2)
        # aa=self.AA_embedding(inputs)
        # embedded_inputs = nn.Linear(512, max(batch_lengths)).cuda()(inputs)  # (1024,256)->(1024,33)
        # Add a sequence dimension
        embedded_inputs1 = inputs.unsqueeze(1).repeat(1, max(batch_lengths), 1)
        embedded_inputs2 = nn.Linear(512, 256).cuda()(embedded_inputs1)
        # input_x = embedded_inputs.view(len(embedded_inputs), 1, -1)#(1024,1,256)
        packed_inputs = pack_padded_sequence(embedded_inputs2, batch_lengths, batch_first=True)

        packed_lstm_out, _ = self.lstm(input=packed_inputs)
        lstm_out, lstm_out_length = pad_packed_sequence(packed_lstm_out, batch_first=True)

        return lstm_out, lstm_out_length

    def _auxiliary_forward(self, x):
        output = self.fc_auxiliary(x)
        return nn.Softmax(dim=1)(output)

    def _annotation_forward(self, annotation_feature):
        vgene, jgene, isotype = annotation_feature

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

def load_model(args=None, antiberty=None):
    if args is None:
        model = BCRSORT(aa_embedding_dim=256, feature_embedding_dim=64, hidden_size=512, num_layers=2, lstm_dropout=0.1,
                        proj_size=0, conv_out_channel=64, kernel_size=[3, 4, 5], kernel_stride=1, conv_dilation=1,
                        fc_dropout=0.5, hidden_fc1=128, hidden_fc2=32, antiberty=antiberty)
    else:
        model = BCRSORT(aa_embedding_dim=args.seq_dim, feature_embedding_dim=args.feature_dim,
                        hidden_size=args.hidden_dim, num_layers=args.layer_lstm, lstm_dropout=args.dropout_lstm,
                        proj_size=args.proj_lstm, conv_out_channel=args.num_channel, kernel_size=args.kernel_size,
                        kernel_stride=args.stride, conv_dilation=args.dilation,
                        fc_dropout=args.dropout_fc, hidden_fc1=args.dim_fc1, hidden_fc2=args.dim_fc2,
                        antiberty=antiberty)

    return model
