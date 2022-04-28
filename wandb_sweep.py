import torch
import torch.optim as optim
import wandb
# from torchvision import datasets, transforms
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
import numpy as np
from dht import DHT_Model
from nn import NeuralNetwork, KinematicChainLoss, Sawtooth
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, ReduceLROnPlateau


def get_dht_model(dht_params):
    return torch.nn.Sequential(
        Sawtooth(-1, 1, -np.pi, np.pi),
        DHT_Model(dht_params)
    )


def get_loss_parameters(config, data_file_A, data_file_B):
    data_A = torch.load(data_file_A)
    data_B = torch.load(data_file_B)

    dht_params_A = data_A["dht_params"]
    joint_limits_A = data_A["joint_limits"]
    dht_model_A = get_dht_model(dht_params_A)

    dht_params_B = data_B["dht_params"]
    joint_limits_B = data_B["joint_limits"]
    dht_model_B = get_dht_model(dht_params_B)

    weight_matrix_positions = False
    weight_matrix_orientations = False

    if config.get("weight_matrix_positions", "") == "auto":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        positions_A = poses_A[:, :3, -1]
        positions_A = torch.cat((torch.zeros(1, 3), positions_A))
        link_lenghts_A = torch.norm(positions_A[1:] - positions_A[:-1], p=2, dim=-1)
        link_lenghts_A = link_lenghts_A.cumsum(0)
        link_lenghts_A = link_lenghts_A / link_lenghts_A[-1]

        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()
        positions_B = poses_B[:, :3, -1]
        positions_B = torch.cat((torch.zeros(1, 3), positions_B))
        link_lenghts_B = torch.norm(positions_B[1:] - positions_B[:-1], p=2, dim=-1)
        link_lenghts_B = link_lenghts_B.cumsum(0)
        link_lenghts_B = link_lenghts_B / link_lenghts_B[-1]

        weight_matrix = (1 - torch.cdist(link_lenghts_A.unsqueeze(-1), link_lenghts_B.unsqueeze(-1))) ** 2
        weight_matrix = weight_matrix * link_lenghts_A.unsqueeze(-1)
        weight_matrix = weight_matrix * link_lenghts_B.unsqueeze(0)

        weight_matrix_positions = weight_matrix
    elif config.get("weight_matrix_positions", "") == "tcp":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros(len(poses_A), len(poses_B))
        weight_matrix[-1, -1] = 1

        weight_matrix_positions = weight_matrix

    if config.get("weight_matrix_orientations", "") == "auto":
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros((len(poses_A), len(poses_B)))
        weight_matrix[-1, -1] = 1.
        weight_matrix_orientations = weight_matrix  # .to(device)

    if any([v is False for v in config.values()]):
        poses_A = dht_model_A(torch.zeros((1, len(joint_limits_A)))).squeeze()
        poses_B = dht_model_B(torch.zeros((1, len(joint_limits_B)))).squeeze()

        weight_matrix = torch.zeros((len(poses_A), len(poses_B)))

        if config.get("weight_matrix_positions", "") is False:
            weight_matrix_positions = weight_matrix
        if config.get("weight_matrix_orientations", "") is False:
            weight_matrix_orientations = weight_matrix

    return weight_matrix_positions, weight_matrix_orientations


def train(config=None, project=None):
    # Initialize a new wandb run
    with wandb.init(config=config, project=project):
        # If called by wandb.agent, as below,
        # this config will be set by Sweep Controller
        config = wandb.config

        loader_train = build_dataset(config.data_file_A, config.batch_size)
        loader_test = build_dataset(config.data_file_A, -1, train=False)
        network = build_network(config.data_file_A, config.data_file_B, config.network_width, config.network_depth,
                                config.dropout)
        optimizer = build_optimizer(network, config.optimizer, config.learning_rate)

        scheduler = config.scheduler

        if scheduler == "coswr":
            scheduler = CosineAnnealingWarmRestarts(optimizer, 2_000)
        elif scheduler == "reduce_plateau":
            scheduler = ReduceLROnPlateau(optimizer, factor=.5, patience=150)

        else:
            scheduler = None

        config_loss = {
            "weight_matrix_positions": config.weight_matrix_positions,
            "weight_matrix_orientations": config.weight_matrix_orientations,
        }

        loss_fn_train = KinematicChainLoss(
            *get_loss_parameters(config_loss, config.data_file_A, config.data_file_B)).to(
            device)

        def loss_fn_test(output, target):
            return torch.norm(output[:, -1, :3, -1] - target[:, -1, :3, -1], p=2, dim=-1).sum()

        for epoch in tqdm(range(config.epochs)):
            avg_loss, avg_loss_pos, avg_loss_ori = train_epoch(network, loader_train, optimizer, loss_fn_train)
            avg_loss_test = test_epoch(network, loader_test, loss_fn_test)
            wandb.log({"train loss": avg_loss,
                       "train loss position": avg_loss_pos,
                       "train loss orientation": avg_loss_ori,
                       "test loss": avg_loss_test,
                       "learnin rate": optimizer.param_groups[0]["lr"],
                       "epoch": epoch})

            if scheduler:
                if type(scheduler) == ReduceLROnPlateau:
                    scheduler.step(avg_loss_test)
                else:
                    scheduler.step()


def build_dataset(data_file, batch_size=-1, train=True):
    data = torch.load(data_file)
    dht_params = data["dht_params"]
    dht_model = get_dht_model(dht_params)

    if train:
        X = data["X_train"]
    else:
        X = data["X_test"]

    y = dht_model(X).detach()

    if batch_size == -1:
        batch_size = len(X)

    loader = DataLoader(TensorDataset(X, y), batch_size=batch_size, shuffle=True)

    return loader


def build_network(data_file_A, data_file_B, network_width, network_depth, dropout):
    data_A = torch.load(data_file_A)
    data_B = torch.load(data_file_B)

    dht_params_B = data_B["dht_params"]
    dht_model_B = get_dht_model(dht_params_B)

    network_structure = [('linear', network_width), ('relu', None), ('dropout', dropout)] * network_depth
    network_structure.append(('linear', len(data_B["joint_limits"])))

    def init_weights(m):
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

    state_mapper = NeuralNetwork(len(data_A["joint_limits"]), network_structure)
    state_mapper.apply(init_weights)

    network = torch.nn.Sequential(
        state_mapper,
        dht_model_B
    )

    if torch.cuda.device_count() > 1:
        network = torch.nn.DataParallel(network)

    return network.to(device)


def build_optimizer(network, optimizer, learning_rate):
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer


def test_epoch(network, loader, loss_fn):
    network.eval()
    cumu_loss = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)

        # ➡ Forward pass
        cumu_loss += loss_fn(network(data), target)

    cumu_loss /= len(loader.dataset)

    return cumu_loss


def train_epoch(network, loader, optimizer, loss_fn):
    network.train()
    cumu_loss = 0
    cumu_loss_position = 0
    cumu_loss_orientation = 0
    for _, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # ➡ Forward pass
        loss, loss_position, loss_orientation = loss_fn(network(data), target)
        cumu_loss += loss.item()
        cumu_loss_position += loss_position.item()
        cumu_loss_orientation += loss_orientation.item()

        # ⬅ Backward pass + weight update
        loss.backward()
        optimizer.step()

        # wandb.log({"batch loss": loss.item()})

    cumu_loss /= len(loader.dataset)
    cumu_loss_position /= len(loader.dataset)
    cumu_loss_orientation /= len(loader.dataset)

    return cumu_loss, cumu_loss_position, cumu_loss_orientation


if __name__ == '__main__':
    train({
        "data_file_A": "data/panda_10000_1000.pt",
        "data_file_B": "data/ur5_10000_1000.pt",

        'epochs': 5_000,
        'batch_size': 32,
        'optimizer': 'adam',
        'learning_rate': 1e-3,
        'scheduler': 'reduce_plateau',
        'network_depth': 8,
        'network_width': 512,
        'dropout': 0.1,
        'weight_matrix_positions': 'auto',
        'weight_matrix_orientations': False,
    }, project="robot2robot_state_mapper")

    exit()
    wandb.login()

    ## WAND Setup
    sweep_config = {
        'method': 'random',
        'metric': {
            'name': 'test loss',
            'goal': 'minimize'
        },
        'parameters': {
            "data_file_A": {'value': "data/panda_10000_1000.pt"},
            "data_file_B": {'value': "data/ur5_10000_1000.pt"},

            'epochs': {
                'value': 5_000
            },
            'batch_size': {
                'values': [16, 32, 64, 128, 256, 512]
            },
            'optimizer': {
                'values': ['adam', 'sgd']
            },
            'learning_rate': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1
            },
            'network_depth': {
                'values': [4, 8, 16]
            },
            'network_width': {
                'values': [64, 256, 512, 1024]
            },
            'dropout': {
                'values': [0.1, 0.2]
            },
            'weight_matrix_positions': {
                'value': 'tcp'
            },
            'weight_matrix_orientations': {
                'value': False
            }
        }
    }

    sweep_id = wandb.sweep(sweep_config, project="robot2robot_state_mapper")

    import pprint

    pprint.pprint(sweep_config)
    wandb.agent(sweep_id, train, count=20)
