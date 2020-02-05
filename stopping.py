import math

class EarlyStoppingCallback:

    def __init__(self, patience):
        # initialize all members you need
        self.patience = patience
        self.hits = float("Inf")  # Initialize the lowest loss as infinite large
        self.epoch_num = None

    def step(self, current_loss):
        # check whether the current loss is lower than the previous best value.
        # if not count up for how long there was no progress
        if current_loss < self.hits:
            self.hits = current_loss
            self.epoch_num = 0
        else:
            self.epoch_num += 1

    def should_stop(self):
        # check whether the duration of where there was no progress is larger or equal to the patience
        if self.epoch_num >= self.patience:
            return True
        else:
            return False


