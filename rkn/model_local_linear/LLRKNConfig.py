from model.RKNConfig import RKNConfig

class LLRKNConfig(RKNConfig):


    @property
    def num_basis(self):
        raise NotImplementedError("Num Bases Matrices not given")

    @property
    def transition_network_hidden_dict(self):
        raise NotImplementedError("Transition Network hidden Dict not given")