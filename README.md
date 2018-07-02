# A Simple Neural Attentive Meta-Learner (SNAIL) in PyTorch
An implementation of Simple Neural Attentive Meta-Learner (SNAIL) ([paper](https://arxiv.org/pdf/1707.03141.pdf)) in PyTorch.

Much of the boiler plate code for setting up datasets and what not came from a PyTorch implementation of [Prototypical Networks](https://github.com/orobix/Prototypical-Networks-for-Few-shot-Learning-PyTorch/blob/master/README.md).

## Mini-Imagenet Dataset

Follow the instructions here: https://github.com/renmengye/few-shot-ssl-public
to download the mini-imagenet dataset.

## Performance

Below are the following attempts to reproduce the results in the reference
paper:

### Omniglot:

| Model | 1-shot (5-way Acc.) | 5-shot (5-way Acc.) | 1 -shot (20-way Acc.) | 5-shot (20-way Acc.)|
| --- | --- | --- | --- | --- |
| Reference Paper | 99.07% | 99.78% | 97.64% | 99.36%|
| This repo | 98.31%\* | 99.26%\*\* | 93.75%° | 97.88%°° |

\* achieved running `python train.py --exp omniglot_5way_1shot --cuda`

\* achieved running `python train.py --exp omniglot_5way_5shot --num_samples 5 --cuda`

\* achieved running `python train.py --exp omniglot_20way_1shot --num_cls 20 --cuda`

\* achieved running `python train.py --exp omniglot_20way_5shot --num_cls 20
--num_samples 5 --cuda`

### Mini-Imagenet:

In progress. Writing the code for the experiments should be done soon but the
main bottleneck in these experiments for me is compute, if someone would be
willing to run and report numbers that would be much appreciated.

### RL:

In progress.
