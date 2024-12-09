import torch
from torch import nn
from util.misc import NestedTensor
from typing import List
from torch.autograd import Function
import torch.nn.functional as F
import math
import copy


def lambda_(p, gamma = 0.05):
    return (2/(math.exp(-gamma*p)+1)) - 1


class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None


class DomainClassifier(nn.Module):
    def __init__(self, feature_dims, modality_map_dim, gamma: float = 0.15, iter_per_epoch: int = 2000, hidden_dim: int = 512):
        super(DomainClassifier, self).__init__()
        input_dim = feature_dims[0][0]
        mini_modality_map_dim = modality_map_dim // (feature_dims[0][1] * feature_dims[0][2])
        
        self.classifier = nn.Sequential(
                                        nn.BatchNorm1d(input_dim),
                                        nn.Linear(input_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, hidden_dim),
                                        nn.BatchNorm1d(hidden_dim),
                                        nn.ReLU(),
                                        nn.Linear(hidden_dim, mini_modality_map_dim),
                                      )
        
        self.gamma = gamma
        self.epoch = 0

    def update_lambda(self):
        if self.training:
            self.lambd = lambda_(self.epoch, self.gamma) if self.gamma != 0 else 0

    def forward(self, x):
        self.update_lambda()

        x0 = x[0].tensors.movedim(1, -1)
        x0 = x0.reshape(-1, x0.shape[-1])
        x0_reversed = ReverseLayerF.apply(x0, self.lambd)

        return self.classifier(x0_reversed)


class FeatureInvarianceWrapper(nn.Module):
    def __init__(self, dino: nn.Module, patch_size: int = 4, im_dim = [3, 1024, 1280], dataset: str = '', gamma: float = 0.15):
        super().__init__()
        if dataset == 'flir':
            im_dim = [3, 640, 512]
        self.dino = dino
        dummy_input = torch.zeros([1] + im_dim)
        feature_dims = self.dino.backbone[0].forward_raw(dummy_input)
        feature_dims = [[f.shape[1], int(f.shape[2]), int(f.shape[3])] for f in feature_dims]
                
        # feature_dims = [[f.shape[1], int(f.shape[2]//2), int(f.shape[3]//2)] for f in feature_dims] #TODO Maybe add this option if dilation=True
        _, _, py, px = self.dino.backbone[0].patch_embed(dummy_input).shape
        modality_map_dim = py * px
        self.domain_classifier = DomainClassifier(feature_dims, modality_map_dim, gamma=gamma)

    def forward(self, samples: NestedTensor, targets: List = None):
        # Feature extraction
        features, poss = self.dino.backbone_forward(samples)

        # Detection Head
        out = self.dino.detection_forward(samples, features, poss, targets)
        
        if self.training:
            target_modality_map = features[2].modality_map
            out['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
            # Domain Classification
            out['output_modality_map'] = self.domain_classifier(features)

        return out
    
    def classifier_forward(self, samples: NestedTensor, targets: List = None):
        # Feature extraction
        features, poss = self.dino.backbone_forward(samples)
        target_modality_map = features[2].modality_map

        # Detection Head
        out = {}
        out['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
        # Domain Classification
        o = self.domain_classifier(features)
        o[o>0.5] = 1
        o[o<=0.5] = 0
        out['output_modality_map'] = o

        return out
    
    def map_resize(self, modality_map, feature_map):
        _, mmh, mmw = modality_map.shape
        _, _, fmh, fmw = feature_map.shape
        mh, mw = mmh//fmh, mmw//fmw

        return modality_map.unfold(1, mh, mw).unfold(2, mh, mw).reshape(-1, mh * mw)

    def train_classifier_only_forward(self, samples: NestedTensor, targets: List = None):
        # Feature extraction
        with torch.no_grad():
            features, poss = self.dino.backbone_forward(samples)
            target_modality_map = features[2].modality_map

        # Detection Head
        out = {}
        out['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
        # Domain Classification
        out['output_modality_map'] = self.domain_classifier(features)

        return out
    
class FeatureInvarianceTrainer(nn.Module):
    def __init__(self, backbone: nn.Module, patch_size: int = 4, im_dim = [3, 1024, 1280], dataset: str = '', gamma: float = 0.15):
        super().__init__()
        if dataset == 'flir':
            im_dim = [3, 640, 512]
        dummy_input = torch.zeros([1] + im_dim)
        feature_dims = backbone.forward_raw(dummy_input)
        feature_dims = [[f.shape[1], int(f.shape[2]), int(f.shape[3])] for f in feature_dims]
                
        # feature_dims = [[f.shape[1], int(f.shape[2]//2), int(f.shape[3]//2)] for f in feature_dims] #TODO Maybe add this option if dilation=True
        _, _, py, px = backbone.patch_embed(dummy_input).shape
        modality_map_dim = py * px
        self.domain_classifier = DomainClassifier(feature_dims, modality_map_dim, gamma=gamma)

    def forward(self, features, out_dino):
        if self.training:
            target_modality_map = features[2].modality_map
            out_dino['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
            # Domain Classification
            out_dino['output_modality_map'] = self.domain_classifier(features)

        return out_dino
    
    def classifier_forward(self, samples: NestedTensor, targets: List = None):
        # Feature extraction
        features, poss = self.dino.backbone_forward(samples)
        target_modality_map = features[2].modality_map

        # Detection Head
        out = {}
        out['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
        # Domain Classification
        o = self.domain_classifier(features)
        o[o>0.5] = 1
        o[o<=0.5] = 0
        out['output_modality_map'] = o

        return out
    
    def map_resize(self, modality_map, feature_map):
        _, mmh, mmw = modality_map.shape
        _, _, fmh, fmw = feature_map.shape
        mh, mw = mmh//fmh, mmw//fmw

        return modality_map.unfold(1, mh, mw).unfold(2, mh, mw).reshape(-1, mh * mw)

    def train_classifier_only_forward(self, samples: NestedTensor, targets: List = None):
        # Feature extraction
        with torch.no_grad():
            features, poss = self.dino.backbone_forward(samples)
            target_modality_map = features[2].modality_map

        # Detection Head
        out = {}
        out['target_modality_map'] = self.map_resize(target_modality_map, features[0].tensors)
        
        # Domain Classification
        out['output_modality_map'] = self.domain_classifier(features)

        return out
    