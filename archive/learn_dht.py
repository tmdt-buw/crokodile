import datetime
import os
import tempfile
from itertools import chain

import torch
from tensorboard import program
from torch.utils.data import DataLoader, TensorDataset
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm

from dht import DHT_Model, DHT_Transform
from nn import Rescale, WeightedPoseLoss, HomoscedasticLoss

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = "cpu"
save_results = True

if __name__ == '__main__':
    import numpy as np

    import sys
    from pathlib import Path

    sys.path.append(str(Path(__file__).resolve().parents[1]))

    if save_results:
        tb_dir = f"results/{os.path.basename(__file__).replace('.py', '')}"
        results_dir = os.path.join(tb_dir, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    else:
        results_dir = tempfile.NamedTemporaryFile().name
        tb_dir = results_dir

    tb = program.TensorBoard()
    try:
        tb.configure(argv=[None, '--logdir', tb_dir, '--port', '6666'])
        url = tb.launch()
    except:
        tb.configure(argv=[None, '--logdir', tb_dir])
        url = tb.launch()
    print(f"Tensorflow listening on {url}")

    configurations = [
        {
            "epochs": 2_000,
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            # "config_loss": {"weight_orientation": .1},
            "patience": 500
        },

        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "config_loss": {"weight_orientation": 0}
        },
        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "continue_training": True,
            "config_loss": {"weight_orientation": .1}
        },
        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "continue_training": True,
            "config_loss": {"weight_orientation": .5}
        },
        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "continue_training": True,
            "config_loss": {"weight_orientation": 1}
        },

        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "config_loss": {"weight_orientation": .1}
        },
        {
            "batch_size": 512,
            "network_depth": 2,
            "data_file": "data/panda_10000_1000.pt",
            "continue_training": True,
            "config_loss": {"weight_orientation": .5}
        },
        {
            "batch_size": 512,
            "data_file": "data/panda_10000_1000.pt",
            "continue_training": True,
            "config_loss": {"weight_orientation": 1}
        },

    ]

    model = None

    for cid, configuration in enumerate(configurations):
        writer = SummaryWriter(os.path.join(results_dir, str(cid)))

        epochs = configuration.get("epochs", 500)
        batch_size = configuration.get("batch_size", 512)
        network_depth = configuration.get("network_depth", 0)
        lr = configuration.get("lr", 1e-3)
        data_file = configuration["data_file"]
        continue_training = configuration.get("continue_training", False)
        config_loss = configuration.get("config_loss", {})
        patience = configuration.get("patience", None)

        best_performance = np.inf
        steps_since_best = 0

        if torch.cuda.device_count() > 1:
            batch_size *= torch.cuda.device_count()

        data = torch.load(data_file)

        X_train, y_train, X_test, y_test = data["X_train"], data["y_train"], data["X_test"], data["y_test"]
        dht_params = data["dht_params"]
        joint_limits = data["joint_limits"]

        data_loader_train = DataLoader(TensorDataset(X_train, y_train), batch_size=batch_size)

        X_test = X_test.to(device)
        y_test = y_test.to(device)

        if not continue_training or not cid:
            # Option 1: Use known DHT params
            model = torch.nn.Sequential(
                Rescale(X_train.shape[1:]),
                DHT_Model(dht_params, upscale_dim=True)
            )


            # model[1].m.data = (joint_limits[:, 1] - joint_limits[:, 0]) / 2
            # model[1].c.data = (joint_limits[:, 1] - joint_limits[:, 0]) / 2 + joint_limits[:, 0]

            def init_weights(m):
                if isinstance(m, torch.nn.Linear):
                    torch.nn.init.xavier_uniform(m.weight)
                # elif isinstance(m, Rescale):
                #     torch.nn.init.normal_(m.m)
                #     torch.nn.init.normal_(m.c)
                elif isinstance(m, DHT_Transform) or isinstance(m, Rescale):
                    for p in m.parameters():
                        if p.requires_grad:
                            torch.nn.init.normal_(p)


            model.apply(init_weights)

            if torch.cuda.device_count() > 1:
                print("Let's use", torch.cuda.device_count(), "GPUs!")
                model = torch.nn.DataParallel(model)

            model.to(device)

        # else:
        #     # Option 2: Use random DHT_params
        #     dht_params_init = []
        #     for dht_param in dht_params:
        #         dht_param = deepcopy(dht_param)
        #         if "theta" not in dht_param or "d" not in dht_param:
        #             # only consider transformations with a degree of freedom
        #             for randomizeable_param, limits in [("theta", (-np.pi, np.pi)), ("d", (-1, 1)), ("a", (-1, 1)), ("alpha", (-np.pi, np.pi))]:
        #                 if randomizeable_param in dht_param:
        #                     dht_param[randomizeable_param] = np.random.uniform(*limits)
        #
        #             # Optional: Specify if proximal instead of original DHT convention should be used
        #             #dht_param["proximal"] = True
        #             dht_params_init.append(dht_param)
        #     # add one constant transformation for potential tcp offset
        #     dht_params_init.append({param: np.random.uniform(*limits) for param, limits in [("theta", (-np.pi, np.pi)), ("d", (-1, 1)), ("a", (-1, 1)), ("alpha", (-np.pi, np.pi))]})

        # loss_function = WeightedPoseLoss(**config_loss).to(device)
        loss_function = HomoscedasticLoss().to(device)
        optimizer = torch.optim.AdamW(chain(model.parameters(), loss_function.parameters()), lr=lr)

        for epoch in tqdm(range(epochs)):
            model.train()

            for x, y in data_loader_train:
                x = x.to(device)
                y = y.to(device)

                prediction_poses = model(x)

                optimizer.zero_grad()
                loss, _, _ = loss_function(prediction_poses, y)

                loss.backward()
                optimizer.step()

            with torch.no_grad():
                model.eval()
                prediction_poses = model(X_test)
                loss_test, loss_position, loss_orientation = loss_function(prediction_poses, y_test)

                writer.add_scalar(f'test loss', loss_test.item(), epoch)
                writer.add_scalar(f'test loss position', loss_position.item(), epoch)
                writer.add_scalar(f'test loss orientation', loss_orientation.item(), epoch)
                writer.flush()

                if loss_test.item() < best_performance:
                    # log_path = os.path.join(results_dir, str(cid), f"model_{loss_test.item():.3f}.pt")
                    log_path = os.path.join(results_dir, str(cid), f"model.pt")
                    if isinstance(model, torch.nn.DataParallel):
                        torch.save(model.module.state_dict(), log_path)
                    else:
                        torch.save(model.state_dict(), log_path)
                    best_performance = loss_test.item()
                    steps_since_best = 0
                elif patience is not None:
                    steps_since_best += 1
                    if steps_since_best > patience:
                        break

