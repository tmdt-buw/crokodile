from trainer import Trainer


class Expert(Trainer):
    def __init__(self, config):
        self.model_cls = config["Expert"]["model_cls"]
        self.model_config = config["Expert"]["model"]
        self.model_config.update(config["EnvSource"])

        super(Expert, self).__init__(config)

    def generate(self):
        super(Expert, self).generate()
        self.train(**self.config["Expert"]["train"])

    @classmethod
    def get_relevant_config(cls, config):
        return super(Expert, cls).get_relevant_config(config)
