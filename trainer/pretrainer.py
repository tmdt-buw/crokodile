import tempfile


from demonstrations import DemonstrationsTarget
from trainer import Trainer

class Pretrainer(Trainer):
    def __init__(self, config):
        self.model_cls = config["Pretrainer"]["model_cls"]
        self.model_config = config["Pretrainer"]["model"]
        self.model_config.update(config["EnvTarget"])

        super(Pretrainer, self).__init__(config)

    def generate(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            demonstrations = DemonstrationsTarget(self.config)
            demonstrations.save(tmpdir)

            self.model_config.update(
                {
                    "input": tmpdir,
                    "input_config": {
                        "format": "json",
                        "postprocess_inputs": False,
                    },
                }
            )

            super(Pretrainer, self).generate()
            self.train(**self.config["Pretrainer"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(Pretrainer, cls).get_relevant_config(config),
            **DemonstrationsTarget.get_relevant_config(config),
        }
