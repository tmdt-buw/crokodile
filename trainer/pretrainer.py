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
        # Create temporary directory to store demonstrations.
        # Unfortunately, we can't use tempfile.TemporaryDirectory() as a context manager
        # because the rllib model in self.train blocks the *.json files.
        # This leads to a "PermissionError" when the context closes.
        # This is fixed in Python > 3.10. by using TemporaryDirectory(ignore_cleanup_errors=True)
        # For now, we apply the workaround from
        # https://www.scivision.dev/python-tempfile-permission-error-windows/
        tmpdir_ = tempfile.TemporaryDirectory()
        tmpdir = tmpdir_.name

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

        try:
            tmpdir_.cleanup()
        except (NotADirectoryError, PermissionError):
            # temporary directory remains until OS cleans up
            pass

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(Pretrainer, cls).get_relevant_config(config),
            **DemonstrationsTarget.get_relevant_config(config),
        }
