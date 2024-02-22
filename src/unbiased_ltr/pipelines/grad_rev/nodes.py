from functools import partial, update_wrapper

from common_nodes import train_two_tower

train = update_wrapper(
    partial(train_two_tower, dropout_prob=-1),
    train_two_tower,
)
