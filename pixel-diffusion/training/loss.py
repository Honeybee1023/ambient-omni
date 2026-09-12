import torch
from torch_utils import persistence
import ambient_utils
import math
from scipy.stats import truncnorm
import numpy as np


#----------------------------------------------------------------------------

# Ambient-o EDM Loss

@persistence.persistent_class
class AmbientEDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5, *args, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, x_tn, sigma_tn, sigma_t, labels=None, augment_labels=None, *args, **kwargs):
        try:
            net._set_static_graph()
        except:
            pass

        sigma_tn = sigma_tn[:, None, None, None]
        sigma_t = sigma_t[:, None, None, None]

        # add additional noise to reach the level sigma
        noise_tn_to_t = torch.randn_like(x_tn) * torch.sqrt(sigma_t ** 2 - sigma_tn ** 2)
        x_t = x_tn + noise_tn_to_t

        x0_pred = net(x_t, sigma_t, labels, augment_labels=augment_labels)        
        x_tn_pred = ambient_utils.from_x0_pred_to_xnature_pred_ve_to_ve(x0_pred, x_t, sigma_t, sigma_tn)

        edm_weight = (self.sigma_data ** 2 + sigma_t ** 2) / (sigma_t ** 2 * self.sigma_data ** 2)
        ambient_factor = sigma_t ** 4 / ((sigma_t ** 2 - sigma_tn ** 2) ** 2)
        ambient_weight = edm_weight * ambient_factor

        loss = ambient_weight * ((x_tn_pred - x_tn) ** 2)
        return loss, x0_pred, sigma_t, x_t

#----------------------------------------------------------------------------

# Ambient-o classifier loss

@persistence.persistent_class
class AmbientEDMCLSLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5,
                 cls_sigma_pmean=None, cls_sigma_pstd=None, *args, **kwargs):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data
        # Optional: draw the classifier's training noise from its own
        # log-normal instead of the sampler's (P_mean -1.2, P_std 1.2). Off by
        # default. Exists because a corruption as mild as a sigma=0.5 blur
        # (~5/255) is invisible under the standard distribution's median noise
        # (~38/255): only ~5% of samples carry the cue and the classifier sits
        # at chance. Shifting P_mean to ~-3 puts the cue inside most samples.
        # A deviation from the Ambient-o recipe; document it wherever used.
        self.cls_sigma_pmean = cls_sigma_pmean
        self.cls_sigma_pstd = cls_sigma_pstd

    def __call__(self, net, x0, sigma_t, cls_labels, augment_labels=None, labels=None, *args, **kwargs):
        net._set_static_graph()
        if self.cls_sigma_pmean is not None:
            pstd = self.cls_sigma_pstd if self.cls_sigma_pstd is not None else self.P_std
            sigma_t = (torch.randn_like(sigma_t) * pstd + self.cls_sigma_pmean).exp()
        sigma_t = sigma_t[:, None, None, None]    
        
        x_t = x0 + torch.randn_like(x0) * sigma_t
        output = net(x_t, sigma_t, labels, augment_labels=augment_labels)

        x0_pred = output["x0_pred"]
        cls_logits = output["cls_logits"]

        loss = torch.nn.functional.binary_cross_entropy_with_logits(cls_logits, cls_labels.float(), reduction='none')

        return loss, x0_pred, sigma_t, x_t
