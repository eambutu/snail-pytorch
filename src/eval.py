import argparse
import torch
from snail import SnailFewShot
from utils import init_dataset
from train import batch_for_few_shot

parser = argparse.ArgumentParser()
parser.add_argument('--exp', type=str, default='default')
parser.add_argument('--iterations', type=int, default=10000)
parser.add_argument('--dataset', type=str, default='mini_imagenet')
parser.add_argument('--num_cls', type=int, default=5)
parser.add_argument('--num_samples', type=int, default=1)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--cuda', action='store_true')
options = parser.parse_args()

model = SnailFewShot(5, 1, 'mini_imagenet')
weights = torch.load('mini_imagenet_5way_1shot/best_model.pth')
model.load_state_dict(weights)
model = model.cuda()
_, val_dataloader, _, _ = init_dataset(options)
val_iter = iter(val_dataloader)
for batch in val_iter:
    x, y = batch
    x, y, last_targets = batch_for_few_shot(options, x, y)
    model_output = model(x, y)
    last_model = model_output[:, -1, :]

    print(last_model)
    print(last_targets)
    import pdb; pdb.set_trace()
