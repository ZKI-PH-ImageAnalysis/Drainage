import math

import torch
import torch.nn.functional as F


class MeanAbsoluteError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(MeanAbsoluteError, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred =  F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mae = 1. - torch.sum(label_one_hot * pred, dim=1)
        return mae.mean()

class MeanSquareError(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.mse = torch.nn.MSELoss()
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        mse = self.mse(pred, label_one_hot)
        return mse

    
class CrossEntropy(torch.nn.Module):
    def __init__(self, delta=0) -> None:
        super().__init__()
        self.ce = torch.nn.CrossEntropyLoss()
        self.delta = delta

    def forward(self, pred, labels, model=None):
        ce = self.ce(pred, labels)
        if model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            ce = ce + self.delta * l1_norm
        
        return ce

class RevserseCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1 * torch.sum(pred * torch.log(label_one_hot), dim=1))
        return rce.mean()

class GeneralizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, q=0.7, delta=0.0) -> None:
        super(GeneralizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
        self.q = q
        self.delta = delta

    def forward(self, pred, labels, model = None):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        gce = (1. - torch.pow(torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        
        if self.delta != 0.0 and model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            gce = gce + self.delta * l1_norm
        return gce.mean()

class SymmetricCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, delta=0.0) -> None:
        super(SymmetricCrossEntropy, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.num_classes = num_classes
        self.delta = delta
        self.cross_entropy = torch.nn.CrossEntropyLoss()

    def forward(self, pred, labels, model=None):
        # CCE
        ce = self.cross_entropy(pred, labels)

        # RCE
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-7, max=1.0)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        label_one_hot = torch.clamp(label_one_hot, min=1e-4, max=1.0)
        rce = (-1*torch.sum(pred * torch.log(label_one_hot), dim=1))

        # Loss
        loss = self.alpha * ce + self.beta * rce.mean()
        
        if self.delta != 0.0 and model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.delta * l1_norm
        
        return loss

class NormalizedCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes) -> None:
        super(NormalizedCrossEntropy, self).__init__()
        self.num_classes = num_classes
    
    def forward(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
class NormalizedNegativeCrossEntropy(torch.nn.Module):
    def __init__(self, num_classes, min_prob) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.A = - torch.tensor(min_prob).log()
    
    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()

class FocalLoss(torch.nn.Module):

    def __init__(self, gamma=0, alpha=None, size_average=True):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.Tensor([alpha, 1-alpha])
        if isinstance(alpha, list):
            self.alpha = torch.Tensor(alpha)
        self.size_average = size_average

    def forward(self, input, target):
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)                         # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))    # N,H*W,C => N*H*W,C
        target = target.view(-1, 1)

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())

        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)
            at = self.alpha.gather(0, target.data.view(-1))
            logpt = logpt * torch.autograd.Variable(at)

        loss = -1 * (1-pt)**self.gamma * logpt
        if self.size_average:
            return loss.mean()
        else:
            return loss.sum()

class NormalizedFocalLoss(torch.nn.Module):
    def __init__(self, gamma=0, num_classes=10):
        super(NormalizedFocalLoss, self).__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        
    def forward(self, input, target):
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = loss / normalizor

        return loss.mean()

class NormalizedNegativeFocalLoss(torch.nn.Module):
    def __init__(self, num_classes, gamma, min_prob=1e-7) -> None:
        super().__init__()
        self.gamma = gamma
        self.num_classes = num_classes
        self.min_prob = min_prob
        self.logmp = torch.tensor(self.min_prob).log()
        self.A = - (1 - min_prob)**gamma * self.logmp
    
    def forward(self, input, target):
        logmp = self.logmp.to(input.device)
        target = target.view(-1, 1)
        logpt = F.log_softmax(input, dim=1).clamp(min=logmp)
        normalizor = torch.sum(-1 * (1 - logpt.data.exp()) ** self.gamma * logpt, dim=1)
        logpt = logpt.gather(1, target)
        logpt = logpt.view(-1)
        pt = torch.autograd.Variable(logpt.data.exp())
        loss = -1 * (1-pt)**self.gamma * logpt
        loss = 1 - (self.A - loss) / (self.num_classes * self.A - normalizor)
        return loss.mean()

class AGCELoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1, q=2):
        super(AGCELoss, self).__init__()
        self.a = a
        self.q = q
        self.num_classes = num_classes

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = ((self.a+1)**self.q - torch.pow(self.a + torch.sum(label_one_hot * pred, dim=1), self.q)) / self.q
        return loss.mean()

class AUELoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1.5, q=0.9):
        super(AUELoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.q = q

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = (torch.pow(self.a - torch.sum(label_one_hot * pred, dim=1), self.q) - (self.a-1)**self.q)/ self.q
        return loss.mean()

class ANormLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=1.5, p=0.9):
        super(ANormLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a
        self.p = p

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = torch.clamp(pred, min=1e-5, max=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.sum(torch.pow(torch.abs(self.a * label_one_hot-pred), self.p), dim=1) - (self.a-1)**self.p
        return loss.mean() / self.p

class AExpLoss(torch.nn.Module):
    def __init__(self, num_classes=10, a=3):
        super(AExpLoss, self).__init__()
        self.num_classes = num_classes
        self.a = a

    def forward(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        loss = torch.exp(-torch.sum(label_one_hot * pred, dim=1) / self.a)
        return loss.mean()

class ActivePassiveLoss(torch.nn.Module):
    def __init__(self, active_loss, passive_loss,
                 alpha=1., beta=1., delta=0.) -> None:
        super(ActivePassiveLoss, self).__init__()
        self.active_loss = active_loss
        self.passive_loss = passive_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        
    def forward(self, pred, labels, model=None):
        loss = self.alpha * self.active_loss(pred, labels) \
            + self.beta * self.passive_loss(pred, labels)
        if self.delta != 0.0 and model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss = loss + self.delta * l1_norm
        return loss

class ActiveNegativeLoss(torch.nn.Module):
    def __init__(self, active_loss, negative_loss,
                 alpha=1., beta=1., delta=0.) -> None:
        super().__init__()
        self.active_loss = active_loss
        self.negative_loss = negative_loss
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
    
    def forward(self, pred, labels, model):
        al = self.active_loss(pred, labels)
        nl = self.negative_loss(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        
        loss = self.alpha * al + self.beta * nl + self.delta * l1_norm
        
        return loss

class ANL_CE_ER(torch.nn.Module):
    def __init__(self, num_classes, alpha, beta, delta, lamb, min_prob=1e-7, *args, **kwargs) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.alpha = alpha
        self.beta = beta
        self.delta = delta
        self.min_prob = min_prob
        self.lamb = lamb
        self.A = - torch.tensor(min_prob).log()

    def forward(self, pred, labels, model, **kwargs):
        loss_nce = self.nce(pred, labels)
        loss_nnce = self.nnce(pred, labels)
        l1_norm = sum(p.abs().sum() for p in model.parameters())
        entropy = self.entropy_reg(pred)
        return self.alpha * loss_nce + self.beta * loss_nnce + self.delta * l1_norm + self.lamb * entropy
    
    def nce(self, pred, labels):
        pred = F.log_softmax(pred, dim=1)
        label_one_hot = F.one_hot(labels, self.num_classes).float().to(pred.device)
        nce = -1 * torch.sum(label_one_hot * pred, dim=1) / (- pred.sum(dim=1))
        return nce.mean()
    
    def nnce(self, pred, labels):
        pred = F.softmax(pred, dim=1)
        pred = pred.clamp(min=self.min_prob, max=1-self.min_prob)
        pred = self.A + pred.log() # - log(1e-7) - (- log(p(k|x)))
        label_one_hot = F.one_hot(labels, self.num_classes).to(pred.device)
        nnce = 1 - (label_one_hot * pred).sum(dim=1) / pred.sum(dim=1)
        return nnce.mean()
    
    def entropy_reg(self, pred):
        prob = F.softmax(pred, dim=1).clamp(min=self.min_prob, max=1-self.min_prob)
        prob_class = prob.sum(dim=0).view(-1) / prob.sum()
        prob_class = prob_class.clamp(min=self.min_prob, max=1-self.min_prob)
        entropy = math.log(self.num_classes) + (prob_class * prob_class.log()).sum()
        return entropy

class AlphaDrainageLoss(torch.nn.Module):
    def __init__(self, alpha=1.0 ,drainage_idx=-1, delta:float=0,reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.drainage_idx = drainage_idx
        #softmax before exponentiation
        self.reduction = reduction
        self.delta = delta
         
    def forward(self, logits, targets, model):
        '''
        Here teh computation of the drainage loss is done based on logsumexp pf three terms:
        L = logsumexp(0, term_1, term_2)

        where
        term_1 = logsumexp(z_d, {z_j}) - z_t + log(alpha)
        term_2 = logsumexp({z_j}) - z_d - log(alpha)
        z_d : logits of drainage class
        z_t : logits of target class
        {z_j} : logits of all other classes except target and drainage
        logits: shape (batch_size, num_classes)
        targets: shape (batch_size,)
        Returns:
            loss: scalar tensor representing the drainage loss
        '''
        batch_size = logits.size(0)
        device = logits.device
        dtype = logits.dtype
        logits_drainage = logits[:, self.drainage_idx]  # z_d
        logits_targets = logits[torch.arange(batch_size), targets]  # z_t

        # first get logits for the set J (indices other than target and drainage)
        mask_J = torch.ones_like(logits, dtype=bool)
        mask_J[torch.arange(batch_size),  targets] = False
        mask_J[torch.arange(batch_size), self.drainage_idx] = False
        logits_J = logits.masked_select(mask_J).view(batch_size, -1) #same shape as logits - 2

        #compute logsumexp for drainage + J as log(exp(z_d) + sum_j exp(z_j))
        logsumexp_d_J = torch.logsumexp(torch.cat([logits_drainage.unsqueeze(1), logits_J], dim=1), dim=1)

        # Compute logsumexp only for J  as log sum_j exp(z_j)
        logsumexp_J = torch.logsumexp(logits_J, dim=1)

        # get logsumexp terms
        term_0 = torch.zeros_like(logits_targets)  # corresponds to 0
        term_1 = logsumexp_d_J - logits_targets + torch.log(torch.tensor(self.alpha, device=device, dtype=dtype))
        term_2 = logsumexp_J - logits_drainage - torch.log(torch.tensor(self.alpha, device=device, dtype=dtype))

        # stack terms and apply logsumexp
        diff = torch.stack([term_0, term_1, term_2], dim=1)
        lse = torch.logsumexp(diff, dim=1)

        # Reduction
        if self.reduction == "mean":
            loss = lse.mean()
        elif self.reduction == "sum":
            loss = lse.sum()
        else:
            loss = lse

        if self.delta != 0.0 and model is not None:
            l1_norm = sum(p.abs().sum() for p in model.parameters())
            loss += self.delta * l1_norm

        return loss

# Help Function

def _apl(active_loss, passive_loss, config):
    return ActivePassiveLoss(active_loss,
                             passive_loss,
                             config['alpha'],
                             config['beta'],
                             config.get('delta', 0.0))

def _anl(active_loss, negative_loss, config):
    return ActiveNegativeLoss(active_loss,
                              negative_loss,
                              config['alpha'],
                              config['beta'],
                              config['delta'])

# Loss

def mae(num_classes):
    return MeanAbsoluteError(num_classes)

def ce(config):
    delta = config['delta']
    return CrossEntropy(delta=delta)

def rce(num_classes):
    return RevserseCrossEntropy(num_classes)

def nce(num_classes):
    return NormalizedCrossEntropy(num_classes)

def sce(num_classes, config):
    return SymmetricCrossEntropy(num_classes, config['alpha'], config['beta'] ,config.get('delta', 0.0))

def gce(num_classes, config):
    return GeneralizedCrossEntropy(num_classes, config['q'], config.get('delta', 0.0))

def fl(config):
    return FocalLoss(gamma=config['gamma'])

def nfl(num_classes, config):
    return NormalizedFocalLoss(config['gamma'], num_classes)

def nnfl(num_classes, config):
    return NormalizedNegativeFocalLoss(num_classes, config['gamma'], config['min_prob'])

def nnce(num_classes, config):
    return NormalizedNegativeCrossEntropy(num_classes, config['min_prob'])

def agce(num_classes, config):
    return AGCELoss(num_classes, config['a'], config['q'])

def aul(num_classes, config):
    return AUELoss(num_classes, config['a'], config['q'])

def ael(num_classes, config):
    return AExpLoss(num_classes, config['a'])

# Active Passive Loss

def nce_mae(num_classes, config):
    return _apl(nce(num_classes), mae(num_classes), config)

def nce_rce(num_classes, config):
    return _apl(nce(num_classes), rce(num_classes), config)

def nfl_mae(num_classes, config):
    return _apl(nfl(num_classes, config), mae(num_classes), config)

def nfl_rce(num_classes, config):
    return _apl(nfl(num_classes, config), rce(num_classes), config)

# Asymmetric Loss

def nce_agce(num_classes, config):
    return _apl(nce(num_classes), agce(num_classes, config), config)

def nce_aul(num_classes, config):
    return _apl(nce(num_classes), aul(num_classes, config), config)

def nce_ael(num_classes, config):
    return _apl(nce(num_classes), ael(num_classes, config), config)

# Active Negative Loss

def anl_ce(num_classes, config):
    return _anl(nce(num_classes), nnce(num_classes, config), config)

def anl_fl(num_classes, config):
    return _anl(nfl(num_classes, config), nnfl(num_classes, config), config)

# Active Negative Loss with Entropy Regularization
def anl_ce_er(num_classes, config):
    return ANL_CE_ER(num_classes, config['alpha'], config['beta'],
                     config['delta'], config['lamb'], config['min_prob'])
    
def alpha_dl(config):
    return AlphaDrainageLoss(config.get('alpha'), config.get('drainage_idx'), config.get('delta'),
                             config.get('reduction', 'mean'))
