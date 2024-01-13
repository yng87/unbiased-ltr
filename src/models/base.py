class ModelBase:
    def build(self):
        raise NotImplementedError

    def fit(self, train_dataset, eval_dataset):
        raise NotImplementedError

    def predict(self, eval_dataset):
        raise NotImplementedError
