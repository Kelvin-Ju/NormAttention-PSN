import torch
import os
import torch.nn.functional as F

class Criterion(object):
    def __init__(self, args):
        self.setupNormalCrit(args)

    def setupNormalCrit(self, args):
        print('=> Using {} for criterion normal'.format(args.normal_loss))
        self.normal_loss = args.normal_loss
        self.normal_w = args.normal_w
        if args.normal_loss == 'mse':
            self.n_crit = torch.nn.MSELoss(reduce=False, size_average=False)
        elif args.normal_loss == 'cos':
            self.n_crit = torch.nn.CosineEmbeddingLoss(reduce=False, size_average=False)
            self.att_crit = torch.nn.L1Loss(reduce=False, size_average=False)
        else:
            raise Exception("=> Unknown Criterion '{}'".format(args.normal_loss))
        if args.cuda:
            self.n_crit = self.n_crit.cuda()

    def forward(self, output, attention, target):
        self.h_x = output.size()[2]
        self.w_x = output.size()[3]
        self.r = F.pad(output, (0, 1, 0, 0))[:, :, :, 1:]
        self.l = F.pad(output, (1, 0, 0, 0))[:, :, :, :self.w_x]
        self.t = F.pad(output, (0, 0, 1, 0))[:, :, :self.h_x, :]
        self.b = F.pad(output, (0, 0, 0, 1))[:, :, 1:, :]
        self.outputgrad = torch.pow((self.r - self.l) * 0.5, 2) + torch.pow((self.t - self.b) * 0.5, 2)
        self.h_x1 = target.size()[2]
        self.w_x1 = target.size()[3]
        self.r1 = F.pad(target, (0, 1, 0, 0))[:, :, :, 1:]
        self.l1 = F.pad(target, (1, 0, 0, 0))[:, :, :, :self.w_x1]
        self.t1 = F.pad(target, (0, 0, 1, 0))[:, :, :self.h_x1, :]
        self.b1 = F.pad(target, (0, 0, 0, 1))[:, :, 1:, :]
        self.targetgrad = torch.pow((self.r1 - self.l1) * 0.5, 2) + torch.pow((self.t1 - self.b1) * 0.5, 2)
        if self.normal_loss == 'cos':
            num = target.nelement() // target.shape[1]
            if not hasattr(self, 'flag') or num != self.flag.nelement():
                self.flag = torch.autograd.Variable(target.data.new().resize_(num).fill_(1))
            self.out_reshape = output.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.gt_reshape  = target.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.attgrad1 = torch.cat((attention, attention, attention), 1)
            self.attgrad = self.attgrad1.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.attention_reshape = attention.permute(0, 2, 3, 1).contiguous().view(-1, 1)
            self.counterpart = torch.ones_like(self.attention_reshape) - self.attention_reshape
            self.outputgrad1 = self.outputgrad.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.targetgrad1 = self.targetgrad.permute(0, 2, 3, 1).contiguous().view(-1, 3)
            self.grad = self.att_crit(self.outputgrad1, self.targetgrad1)
            self.loss2 = torch.mul(self.attgrad, self.grad)
            self.loss2 = torch.mean(self.loss2)
            self.loss2 = self.loss2 * 0.875
            self.lossnor = self.n_crit(self.out_reshape, self.gt_reshape, self.flag)
            self.loss_depart = self.lossnor.unsqueeze(1)
            self.loss1 = torch.mul(self.counterpart, self.loss_depart)
            self.loss1 = torch.mean(self.loss1)
            self.loss1 = self.loss1 * 0.125
            self.loss = self.loss1 + self.loss2

        elif self.normal_loss == 'mse':
            self.loss = 1

        out_loss = {'N_loss': self.loss.item()}
        return out_loss

    def backward(self):
        self.loss.backward()

def getOptimizer(args, params):
    print('=> Using %s solver for optimization' % (args.solver))
    if args.solver == 'adam':
        optimizer = torch.optim.Adam(params, args.init_lr, betas=(args.beta_1, args.beta_2))
    elif args.solver == 'sgd':
        optimizer = torch.optim.SGD(params, args.init_lr, momentum=args.momentum)
    else:
        raise Exception("=> Unknown Optimizer %s" % (args.solver))
    return optimizer

def getLrScheduler(args, optimizer):
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, 
            milestones=args.milestones, gamma=args.lr_decay, last_epoch=args.start_epoch-2)
    return scheduler

def loadRecords(path, model, optimizer):
    records = None
    if os.path.isfile(path):
        records = torch.load(path[:-8] + '_rec' + path[-8:])
        optimizer.load_state_dict(records['optimizer'])
        start_epoch = records['epoch'] + 1
        records = records['records']
        print("=> loaded Records")
    else:
        raise Exception("=> no checkpoint found at '{}'".format(path))
    return records, start_epoch

def configOptimizer(args, model):
    records = None
    optimizer = getOptimizer(args, model.parameters())
    if args.resume:
        print("=> Resume loading checkpoint '{}'".format(args.resume))
        records, start_epoch = loadRecords(args.resume, model, optimizer)
        args.start_epoch = start_epoch
    scheduler = getLrScheduler(args, optimizer)
    return optimizer, scheduler, records
