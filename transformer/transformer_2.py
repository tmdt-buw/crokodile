import math

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from tqdm import tqdm


class PositionalEncoding(nn.Module):
    def __init__(self,
                 emb_size: int,
                 dropout: float,
                 maxlen: int = 9):
        super(PositionalEncoding, self).__init__()
        den = torch.exp(- torch.arange(0, emb_size, 2) * math.log(10000) / emb_size)
        pos = torch.arange(0, maxlen).reshape(maxlen, 1)
        pos_embedding = torch.zeros((maxlen, emb_size))
        pos_embedding[:, 0::2] = torch.sin(pos * den)
        pos_embedding[:, 1::2] = torch.cos(pos * den)
        pos_embedding.unsqueeze_(-2)

        self.dropout = nn.Dropout(dropout)
        self.register_buffer('pos_embedding', pos_embedding)

    def forward(self, token_embedding: Tensor):
        return self.dropout(token_embedding + self.pos_embedding[:token_embedding.size(0), :])


# Seq2Seq Network
class Seq2SeqTransformer(nn.Module):
    def __init__(self,
                 num_encoder_layers: int,
                 num_decoder_layers: int,
                 emb_size: int,
                 nhead: int,
                 dim_feedforward: int = 512,
                 dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=emb_size,
                                       nhead=nhead,
                                       num_encoder_layers=num_encoder_layers,
                                       num_decoder_layers=num_decoder_layers,
                                       dim_feedforward=dim_feedforward,
                                       dropout=dropout)
        self.positional_encoding = PositionalEncoding(emb_size, dropout=dropout)

    def forward(self,
                src: Tensor,
                trg: Tensor,
                src_mask: Tensor,
                trg_mask: Tensor,
                src_padding_mask: Tensor,
                trg_padding_mask: Tensor,
                memory_key_padding_mask: Tensor):
        src_emb = self.positional_encoding(src)
        trg_emb = self.positional_encoding(trg)
        outs = self.transformer(src_emb, trg_emb, src_mask, trg_mask, None,
                                src_padding_mask, trg_padding_mask, memory_key_padding_mask)
        return outs

    def encode(self, src: Tensor, src_mask: Tensor):
        return self.transformer.encoder(self.positional_encoding(src), src_mask)

    def decode(self, trg: Tensor, memory: Tensor, trg_mask: Tensor):
        return self.transformer.decoder(self.positional_encoding(trg), memory, trg_mask)


def generate_square_subsequent_mask(sz):
    """Generates an upper-triangular matrix of -inf, with zeros on diag."""
    return torch.triu(torch.full((sz, sz), float("-inf")), diagonal=1)


def create_padding_mask(types: Tensor, pad_tokens: list = []):
    padding_mask = torch.zeros_like(types)

    for pad_token in pad_tokens:
        padding_mask |= types == pad_token

    return padding_mask.type(torch.bool)


if __name__ == '__main__':
    from trajectory_encoder import TrajectoryEncoder, generate_trajectory
    from tensorboard import program
    import datetime
    from torch.utils.tensorboard.writer import SummaryWriter

    results_dir = "debug_transformer/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

    tb = program.TensorBoard()
    tb.configure(argv=[None, '--logdir', results_dir])
    url = tb.launch()
    print(f"Tensorflow listening on {url}")

    writer = SummaryWriter(results_dir)

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Scenario parameters
    MAX_SEQ_LENGTH = 5
    STATE_DIMS = [3, 3]

    TOKENS = {
        "STATE": 0,
        "ACTION": 1,
        "START": 2,
        "DONE": 3,
        "PAD": 4
    }

    # Transformer parameters
    NUM_ENCODER_LAYERS = 3
    NUM_DECODER_LAYERS = 3
    FFN_HID_DIM = 1024
    NHEAD = 8
    EMB_SIZE = 32

    # training parameters
    BATCH_SIZE = 64
    lr = 1e-1

    encoder_structs = [[], []]

    te = TrajectoryEncoder(STATE_DIMS, encoder_structs, EMB_SIZE).to(DEVICE)
    te.train()
    transformer = Seq2SeqTransformer(NUM_ENCODER_LAYERS, NUM_DECODER_LAYERS, EMB_SIZE, NHEAD, FFN_HID_DIM).to(DEVICE)
    transformer.train()

    optimizer_transformer = torch.optim.Adam(transformer.parameters(), lr=lr)
    optimizer_te = torch.optim.Adam(te.parameters(), lr=lr)
    loss_function = torch.nn.MSELoss()


    def r2_loss(output, target):
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - output) ** 2)
        r2 = 1 - ss_res / ss_tot
        return r2


    def generate_trajectory_batch(batch_size):
        # generate trajectory encodings
        trajectory_encoding_batch = []
        types_batch = []

        for _ in range(batch_size):
            types = [TOKENS["START"]]
            # types += [TOKENS["STATE"], TOKENS["ACTION"]] * np.random.randint(1, MAX_SEQ_LENGTH // 2)
            types += [TOKENS["STATE"], TOKENS["ACTION"]] * (MAX_SEQ_LENGTH // 2 - 1)
            types += [TOKENS["STATE"], TOKENS["DONE"]]
            types += [TOKENS["PAD"]] * (MAX_SEQ_LENGTH - len(types))

            types = torch.tensor(types, device=DEVICE)
            trajectory = generate_trajectory(STATE_DIMS + [None, None, None], types, device=DEVICE)

            trajectory_encoding = te(trajectory, types)

            trajectory_encoding_batch.append(trajectory_encoding)
            types_batch.append(types)

        trajectory_encoding_batch = torch.stack(trajectory_encoding_batch)

        trajectory_encoding_batch.transpose_(0, 1)
        types_batch = torch.stack(types_batch)
        return trajectory_encoding_batch, types_batch


    for epoch in tqdm(range(1_000)):
        src_trajectory_encoding_batch, src_types_batch = generate_trajectory_batch(BATCH_SIZE)

        # SCENARIO 1: Learn to copy trajectory
        # trg_trajectory_encoding_batch = src_trajectory_encoding_batch.clone()
        # trg_types_batch = src_types_batch.clone()

        # SCENARIO 2: Invert actions
        trg_trajectory_encoding_batch = src_trajectory_encoding_batch.clone()
        trg_types_batch = src_types_batch.clone()

        types_mask = src_types_batch == TOKENS["ACTION"]
        types_mask = types_mask.transpose(0, 1).unsqueeze(-1).repeat(1, 1, EMB_SIZE)
        trg_trajectory_encoding_batch = torch.where(types_mask, -trg_trajectory_encoding_batch,
                                                    trg_trajectory_encoding_batch)

        # SCENARIO 3: Randomize output (should not be learnable, i.e. should approach R2 = 0)
        # trg_trajectory_encoding_batch = torch.rand_like(src_trajectory_encoding_batch)
        # trg_types_batch = src_types_batch.clone()

        # create meaningful dummy scenario
        # src_trajectory_encoding_batch = torch.zeros_like(src_trajectory_encoding_batch)
        # trg_trajectory_encoding_batch = torch.rand_like(src_trajectory_encoding_batch)

        # remove start token from src
        # src_trajectory_encoding_batch = src_trajectory_encoding_batch[1:]
        # src_types_batch = src_types_batch[:, 1:]

        # trg_trajectory_encoding_batch, trg_types_batch = generate_trajectory_batch(BATCH_SIZE)

        # input sequence until second last item, output sequence from second item
        trg_types_batch = trg_types_batch[:, :-1]
        trg_trajectory_encoding_batch_input = trg_trajectory_encoding_batch[:-1]
        trg_trajectory_encoding_batch_output = trg_trajectory_encoding_batch[:-1]
        # trg_trajectory_encoding_batch_output = trg_trajectory_encoding_batch[1:]
        # trg_trajectory_encoding_batch_output = torch.ones_like(trg_trajectory_encoding_batch_output)

        # src_trajectory_encoding_batch = torch.zeros_like(src_trajectory_encoding_batch)
        # trg_trajectory_encoding_batch_input = torch.zeros_like(trg_trajectory_encoding_batch_input)

        src_mask = generate_square_subsequent_mask(src_types_batch.shape[-1]).to(DEVICE)
        src_padding_mask = create_padding_mask(src_types_batch, [TOKENS["PAD"]]).to(DEVICE)

        trg_padding_mask = create_padding_mask(trg_types_batch, [TOKENS["PAD"]]).to(DEVICE)
        trg_mask = generate_square_subsequent_mask(trg_types_batch.shape[-1]).to(DEVICE)

        # print(src_trajectory_encoding_batch.shape, src_types_batch.shape, src_mask.shape, src_padding_mask.shape)
        # print(trg_trajectory_encoding_batch_input.shape, trg_types_batch.shape, trg_mask.shape, trg_padding_mask.shape)

        # transformation = transformer(src_trajectory_encoding_batch, trg_trajectory_encoding_batch_input, src_mask, trg_mask,
        #                              src_padding_mask, trg_padding_mask, src_padding_mask).to(DEVICE)

        # print(transformation.shape)

        # transformer.to(DEVICE)

        # trg_trajectory_encoding_batch_input.requires_grad = True

        transformation_batch = transformer(src_trajectory_encoding_batch,
                                           trg_trajectory_encoding_batch_input, src_mask, trg_mask,
                                           src_padding_mask, trg_padding_mask, src_padding_mask)

        assert not transformation_batch.isnan().any(), transformation_batch

        loss = loss_function(transformation_batch, trg_trajectory_encoding_batch_output)

        r2_score = r2_loss(transformation_batch, trg_trajectory_encoding_batch_output)

        writer.add_scalar('loss', loss.item(), epoch)
        writer.add_scalar('r2', r2_score.item(), epoch)
        writer.add_histogram('prediction', transformation_batch, epoch)

        # print(src_trajectory_encoding_batch[:5,0,0])
        # print(trg_trajectory_encoding_batch_output[:5,0,0])

        optimizer_transformer.zero_grad()
        optimizer_te.zero_grad()

        loss.backward()

        optimizer_transformer.step()
        optimizer_te.step()
