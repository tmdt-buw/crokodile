from copy import deepcopy


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
