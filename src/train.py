# coding=utf-8
from utils import init_dataset
from omniglot_dataset import OmniglotDataset
from snail import SnailFewShot

import argparse
import torch
from torch.autograd import Variable
import torch.nn as nn
from torch.optim import Adam
import numpy as np
from tqdm import tqdm
import os
import copy


def init_model(opt):
    model = SnailFewShot(opt.num_cls, opt.num_samples, opt.dataset)
    model = model.cuda() if opt.cuda else model
    return model

def save_list_to_file(path, thelist):
    with open(path, 'w') as f:
        for item in thelist:
            f.write("%s\n" % item)

def labels_to_one_hot(opt, labels):
    if opt.cuda:
        labels = labels.cpu()
    labels = labels.numpy()
    unique = np.unique(labels)
    map = {label:idx for idx, label in enumerate(unique)}
    idxs = [map[labels[i]] for i in range(labels.size)]
    one_hot = np.zeros((labels.size, unique.size))
    one_hot[np.arange(labels.size), idxs] = 1
    return one_hot, idxs

def batch_for_few_shot(opt, x, y):
    seq_size = opt.num_cls * opt.num_samples + 1
    one_hots = []
    last_targets = []
    for i in range(opt.batch_size):
        one_hot, idxs = labels_to_one_hot(opt, y[i * seq_size: (i + 1) * seq_size])
        one_hots.append(one_hot)
        last_targets.append(idxs[-1])
    last_targets = Variable(torch.Tensor(last_targets).long())
    one_hots = [torch.Tensor(temp) for temp in one_hots]
    y = torch.cat(one_hots, dim=0)
    x, y = Variable(x), Variable(y)
    if opt.cuda:
        x, y = x.cuda(), y.cuda()
        last_targets = last_targets.cuda()
    return x, y, last_targets

def get_acc(last_model, last_targets):
    _, preds = last_model.max(1)
    acc = torch.eq(preds, last_targets).float().mean()
    return acc.item()

def train(opt, tr_dataloader, model, optim, val_dataloader=None):
    if val_dataloader is None:
        best_state = None
    train_loss = []
    train_acc = []
    val_loss = []
    val_acc = []
    best_acc = 0

    best_model_path = os.path.join(opt.exp, 'best_model.pth')
    last_model_path = os.path.join(opt.exp, 'last_model.pth')

    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(opt.epochs):
        print('=== Epoch: {} ==='.format(epoch))
        tr_iter = iter(tr_dataloader)
        model.train()
        model = model.cuda()
        for batch in tqdm(tr_iter):
            optim.zero_grad()
            x, y = batch
            x, y, last_targets = batch_for_few_shot(opt, x, y)
            model_output = model(x, y)
            last_model = model_output[:, -1, :]
            loss = loss_fn(last_model, last_targets)
            loss.backward()
            optim.step()
            train_loss.append(loss.item())
            train_acc.append(get_acc(last_model, last_targets))
        avg_loss = np.mean(train_loss[-opt.iterations:])
        avg_acc = np.mean(train_acc[-opt.iterations:])
        print('Avg Train Loss: {}, Avg Train Acc: {}'.format(avg_loss, avg_acc))
        if val_dataloader is None:
            continue
        val_iter = iter(val_dataloader)
        model.eval()
        for batch in val_iter:
            x, y = batch
            x, y, last_targets = batch_for_few_shot(opt, x, y)
            model_output = model(x, y)
            last_model = model_output[:, -1, :]
            loss = loss_fn(last_model, last_targets)
            val_loss.append(loss.item())
            val_acc.append(get_acc(last_model, last_targets))
        avg_loss = np.mean(val_loss[-opt.iterations:])
        avg_acc = np.mean(val_acc[-opt.iterations:])
        postfix = ' (Best)' if avg_acc >= best_acc else ' (Best: {})'.format(
            best_acc)
        print('Avg Val Loss: {}, Avg Val Acc: {}{}'.format(
            avg_loss, avg_acc, postfix))
        if avg_acc >= best_acc:
            torch.save(model.state_dict(), best_model_path)
            best_acc = avg_acc
            best_state = model.state_dict()
        for name in ['train_loss', 'train_acc', 'val_loss', 'val_acc']:
            save_list_to_file(os.path.join(opt.exp, name + '.txt'), locals()[name])

    torch.save(model.state_dict(), last_model_path)

    return best_state, best_acc, train_loss, train_acc, val_loss, val_acc


def test(opt, test_dataloader, model):
    avg_acc = list()
    for epoch in range(10):
        test_iter = iter(test_dataloader)
        for batch in test_iter:
            x, y = batch
            x, y, last_targets = batch_for_few_shot(opt, x, y)
            model_output = model(x, y)
            last_model = model_output[:, -1, :]
            avg_acc.append(get_acc(last_model, last_targets))
    avg_acc = np.mean(avg_acc)
    print('Test Acc: {}'.format(avg_acc))

    return avg_acc

def main():
    '''
    Initialize everything and train
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument('--exp', type=str, default='default')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--iterations', type=int, default=10000)
    parser.add_argument('--dataset', type=str, default='omniglot')
    parser.add_argument('--num_cls', type=int, default=5)
    parser.add_argument('--num_samples', type=int, default=1)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--cuda', action='store_true')
    options = parser.parse_args()

    if not os.path.exists(options.exp):
        os.makedirs(options.exp)

    if torch.cuda.is_available() and not options.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")

    tr_dataloader, val_dataloader, trainval_dataloader, test_dataloader = init_dataset(
        options)
    model = init_model(options)
    optim = torch.optim.Adam(params=model.parameters(), lr=options.lr)
    res = train(opt=options,
                tr_dataloader=tr_dataloader,
                val_dataloader=val_dataloader,
                model=model,
                optim=optim)
    best_state, best_acc, train_loss, train_acc, val_loss, val_acc = res
    print('Testing with last model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

    model.load_state_dict(best_state)
    print('Testing with best model..')
    test(opt=options,
         test_dataloader=test_dataloader,
         model=model)

if __name__ == '__main__':
    main()
