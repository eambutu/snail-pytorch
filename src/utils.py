# coding=utf-8
from batch_sampler import BatchSampler
from omniglot_dataset import OmniglotDataset
from mini_imagenet_dataset import MiniImagenetDataset
import torch


def init_dataset(opt):
    '''
    Initialize the datasets, samplers and dataloaders
    '''
    if opt.dataset == 'omniglot':
        train_dataset = OmniglotDataset(mode='train')
        val_dataset = OmniglotDataset(mode='val')
        trainval_dataset = OmniglotDataset(mode='trainval')
        test_dataset = OmniglotDataset(mode='test')
    elif opt.dataset == 'mini_imagenet':
        train_dataset = MiniImagenetDataset(mode='train')
        val_dataset = MiniImagenetDataset(mode='val')
        trainval_dataset = MiniImagenetDataset(mode='val')
        test_dataset = MiniImagenetDataset(mode='test')

    tr_sampler = BatchSampler(labels=train_dataset.y,
                                          classes_per_it=opt.num_cls,
                                          num_samples=opt.num_samples,
                                          iterations=opt.iterations,
                                          batch_size=opt.batch_size)

    val_sampler = BatchSampler(labels=val_dataset.y,
                                           classes_per_it=opt.num_cls,
                                           num_samples=opt.num_samples,
                                           iterations=opt.iterations,
                                           batch_size=opt.batch_size)

    trainval_sampler = BatchSampler(labels=trainval_dataset.y,
                                                classes_per_it=opt.num_cls,
                                                num_samples=opt.num_samples,
                                                iterations=opt.iterations,
                                                batch_size=opt.batch_size)

    test_sampler = BatchSampler(labels=test_dataset.y,
                                            classes_per_it=opt.num_cls,
                                            num_samples=opt.num_samples,
                                            iterations=opt.iterations,
                                            batch_size=opt.batch_size)

    tr_dataloader = torch.utils.data.DataLoader(train_dataset,
                                                batch_sampler=tr_sampler)

    val_dataloader = torch.utils.data.DataLoader(val_dataset,
                                                 batch_sampler=val_sampler)

    trainval_dataloader = torch.utils.data.DataLoader(trainval_dataset,
                                                      batch_sampler=trainval_sampler)

    test_dataloader = torch.utils.data.DataLoader(test_dataset,
                                                  batch_sampler=test_sampler)
    return tr_dataloader, val_dataloader, trainval_dataloader, test_dataloader
