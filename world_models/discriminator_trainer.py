from copy import deepcopy
from lit_models.lit_trainer import LitTrainer


class Discriminator(LitTrainer):
    def __init__(self, config):
        self.model_cls = config["Discriminator"]["model_cls"]
        self.model_config = config["Discriminator"]

        super(Discriminator, self).__init__(config)

    def generate(self):
        super(Discriminator, self).generate()
        self.train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(Discriminator, cls).get_relevant_config(config)


class StateMapper(LitTrainer):
    discriminator = None

    def __init__(self, config):
        self.model_cls = config["StateMapper"]["model_cls"]
        self.model_config = config["StateMapper"]

        super(StateMapper, self).__init__(config)

    def generate(self):
        super(StateMapper, self).generate()
        self.discriminator = Discriminator(self.config)
        self.model.discriminator = deepcopy(self.discriminator.model)
        del self.discriminator
        super(StateMapper, self).train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(StateMapper, cls).get_relevant_config(config)
