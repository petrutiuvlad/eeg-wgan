import torch
import torch.nn as nn

def mean_loss(discriminator_loss, generator_loss):
    return -torch.mean(discriminator_loss) + torch.mean(generator_loss)

def generated_loss(generated_loss):
    return -torch.mean(generated_loss)