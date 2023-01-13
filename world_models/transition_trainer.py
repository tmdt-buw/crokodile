from lit_models.lit_trainer import LitTrainer

class TransitionModel(LitTrainer):
    def __init__(self, config):
        self.model_cls = config["TransitionModel"]["model_cls"]
        self.model_config = config["TransitionModel"]
        super(TransitionModel, self).__init__(config)

    def generate(self):
        super(TransitionModel, self).generate()
        self.train()

    @classmethod
    def get_relevant_config(cls, config):
        return super(TransitionModel, cls).get_relevant_config(config)
