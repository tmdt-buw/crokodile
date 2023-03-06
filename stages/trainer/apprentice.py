from trainer import Trainer

from .pretrainer import Pretrainer


class Apprentice(Trainer):
    def __init__(self, config):
        self.model_cls = config["Apprentice"]["model_cls"]
        self.model_config = config["Apprentice"]["model"]
        self.model_config.update(config["EnvTarget"])

        super(Apprentice, self).__init__(config)

    def generate(self):
        pretrainer = Pretrainer(self.config)
        weights = pretrainer.model.get_weights()
        del pretrainer

        super(Apprentice, self).generate()

        self.model.set_weights(weights)
        del weights
        self.train(**self.config["Apprentice"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return {
            **super(Apprentice, cls).get_relevant_config(config),
            **Pretrainer.get_relevant_config(config),
        }
