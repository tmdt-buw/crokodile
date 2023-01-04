from lit_trainer import TransitionModel, Discriminator, StateMapper

def transition_model_main():
    data_file_A = "panda_5_20000_4000.pt"
    data_file_B = "ur5_5_20000_4000.pt"

    config_A = {
        "wandb_config": {
            "project": "robot2robot",
            "entity": "robot2robot",
        },
        "cache": {
            "mode": "wandb",
            "load": True,
            "save": False,
        },
        "TransitionModel": {
            "model_cls": "transition_model",
            "data": data_file_A,
            "log_suffix": "_A",
            "model": {
                "network_width": 256,
                "network_depth": 4,
                "dropout": 0.0,
                "out_activation": "tanh",
            },
            "train": {
                "max_epochs": 50,
                "batch_size": 2048,
                "lr": 1e-3,
            },
            "callbacks": {
            }

        }
    }

    model = TransitionModel(config_A)
    print("test")
def discriminator_main():
    pass

if __name__ == "__main__":
    transition_model_main()
