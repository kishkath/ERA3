import torch



def allot_device(random_seed_value):
    if torch.cuda.is_available():
        device = "cuda"
        torch.manual_seed(random_seed_value)
    else:
        device = "cpu"
    return device

