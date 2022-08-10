import os

data_folder = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

wandb_config = {
    "project": "PITL",
    "entity": "robot2robot",

    "mode": "disabled"
}
