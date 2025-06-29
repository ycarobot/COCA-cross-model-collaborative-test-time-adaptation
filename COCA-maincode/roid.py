import torch
import torch.nn as nn
import torch.nn.functional as F1
from torch.nn.utils.weight_norm import WeightNorm

from copy import deepcopy

import PIL
import torch
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random

from utils.losses import Entropy, SymmetricCrossEntropy, SoftLikelihoodRatio
from utils.misc import ema_update_model
import torchvision.transforms.functional as F
import torchvision.transforms as transforms
from torchvision.transforms import ColorJitter, Compose, Lambda
from numpy import random


class GaussianNoise(torch.nn.Module):
    def __init__(self, mean=0., std=1.):
        super().__init__()
        self.std = std
        self.mean = mean

    def forward(self, img):
        noise = torch.randn(img.size()) * self.std + self.mean
        noise = noise.to(img.device)
        return img + noise

    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)


class Clip(torch.nn.Module):
    def __init__(self, min_val=0., max_val=1.):
        super().__init__()
        self.min_val = min_val
        self.max_val = max_val

    def forward(self, img):
        return torch.clip(img, self.min_val, self.max_val)

    def __repr__(self):
        return self.__class__.__name__ + '(min_val={0}, max_val={1})'.format(self.min_val, self.max_val)


class ColorJitterPro(ColorJitter):
    """Randomly change the brightness, contrast, saturation, and gamma correction of an image."""

    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0, gamma=0):
        super().__init__(brightness, contrast, saturation, hue)
        self.gamma = self._check_input(gamma, 'gamma')

    @staticmethod
    @torch.jit.unused
    def get_params(brightness, contrast, saturation, hue, gamma):
        """Get a randomized transform to be applied on image.

        Arguments are same as that of __init__.

        Returns:
            Transform which randomly adjusts brightness, contrast and
            saturation in a random order.
        """
        transforms = []

        if brightness is not None:
            brightness_factor = random.uniform(brightness[0], brightness[1])
            transforms.append(Lambda(lambda img: F.adjust_brightness(img, brightness_factor)))

        if contrast is not None:
            contrast_factor = random.uniform(contrast[0], contrast[1])
            transforms.append(Lambda(lambda img: F.adjust_contrast(img, contrast_factor)))

        if saturation is not None:
            saturation_factor = random.uniform(saturation[0], saturation[1])
            transforms.append(Lambda(lambda img: F.adjust_saturation(img, saturation_factor)))

        if hue is not None:
            hue_factor = random.uniform(hue[0], hue[1])
            transforms.append(Lambda(lambda img: F.adjust_hue(img, hue_factor)))

        if gamma is not None:
            gamma_factor = random.uniform(gamma[0], gamma[1])
            transforms.append(Lambda(lambda img: F.adjust_gamma(img, gamma_factor)))

        random.shuffle(transforms)
        transform = Compose(transforms)

        return transform

    def forward(self, img):
        """
        Args:
            img (PIL Image or Tensor): Input image.

        Returns:
            PIL Image or Tensor: Color jittered image.
        """
        fn_idx = torch.randperm(5)
        for fn_id in fn_idx:
            if fn_id == 0 and self.brightness is not None:
                brightness = self.brightness
                brightness_factor = torch.tensor(1.0).uniform_(brightness[0], brightness[1]).item()
                img = F.adjust_brightness(img, brightness_factor)

            if fn_id == 1 and self.contrast is not None:
                contrast = self.contrast
                contrast_factor = torch.tensor(1.0).uniform_(contrast[0], contrast[1]).item()
                img = F.adjust_contrast(img, contrast_factor)

            if fn_id == 2 and self.saturation is not None:
                saturation = self.saturation
                saturation_factor = torch.tensor(1.0).uniform_(saturation[0], saturation[1]).item()
                img = F.adjust_saturation(img, saturation_factor)

            if fn_id == 3 and self.hue is not None:
                hue = self.hue
                hue_factor = torch.tensor(1.0).uniform_(hue[0], hue[1]).item()
                img = F.adjust_hue(img, hue_factor)

            if fn_id == 4 and self.gamma is not None:
                gamma = self.gamma
                gamma_factor = torch.tensor(1.0).uniform_(gamma[0], gamma[1]).item()
                img = img.clamp(1e-8, 1.0)  # to fix Nan values in gradients, which happens when applying gamma
                                            # after contrast
                img = F.adjust_gamma(img, gamma_factor)

        return img

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        format_string += 'brightness={0}'.format(self.brightness)
        format_string += ', contrast={0}'.format(self.contrast)
        format_string += ', saturation={0}'.format(self.saturation)
        format_string += ', hue={0})'.format(self.hue)
        format_string += ', gamma={0})'.format(self.gamma)
        return format_string


def get_tta_transforms(img_size, gaussian_std: float=0.005, soft=False, padding_mode='edge', cotta_augs=True):
    n_pixels = img_size[0] if isinstance(img_size, (list, tuple)) else img_size

    tta_transforms = [
        Clip(0.0, 1.0),
        ColorJitterPro(
            brightness=[0.8, 1.2] if soft else [0.6, 1.4],
            contrast=[0.85, 1.15] if soft else [0.7, 1.3],
            saturation=[0.75, 1.25] if soft else [0.5, 1.5],
            hue=[-0.03, 0.03] if soft else [-0.06, 0.06],
            gamma=[0.85, 1.15] if soft else [0.7, 1.3]
        ),
        transforms.Pad(padding=int(n_pixels / 2), padding_mode=padding_mode),
        transforms.RandomAffine(
            degrees=[-8, 8] if soft else [-15, 15],
            translate=(1/16, 1/16),
            scale=(0.95, 1.05) if soft else (0.9, 1.1),
            shear=None,
            interpolation=PIL.Image.BILINEAR,
            fill=0
        )
    ]
    if cotta_augs:
        tta_transforms += [transforms.GaussianBlur(kernel_size=5, sigma=[0.001, 0.25] if soft else [0.001, 0.5]),
                           transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           GaussianNoise(0, gaussian_std),
                           Clip(0.0, 1.0)]
    else:
        tta_transforms += [transforms.CenterCrop(size=n_pixels),
                           transforms.RandomHorizontalFlip(p=0.5),
                           Clip(0.0, 1.0)]

    return transforms.Compose(tta_transforms)



@torch.no_grad()
def update_model_probs(x_ema, x, momentum=0.9):
    return momentum * x_ema + (1 - momentum) * x


class ROID(nn.Module):
    def __init__(self, args, model, optimizer,num_classes):
        super().__init__()
        self.optimizer=optimizer
        self.model=model
        self.args=args
        self.use_weighting = True
        self.use_prior_correction = True
        self.use_consistency = True
        self.momentum_src = 0.99 
        self.momentum_probs = 0.9
        self.temperature = 1/3
        self.batch_size = args.test_batch_size
        self.class_probs_ema = 1 / num_classes * torch.ones(num_classes).to(self.args.gpu)
        self.tta_transform = get_tta_transforms((224, 224), padding_mode="reflect", cotta_augs=False)
        self.device=args.gpu
        # setup loss functions
        self.slr = SoftLikelihoodRatio()
        self.symmetric_cross_entropy = SymmetricCrossEntropy()
        self.softmax_entropy = Entropy()  # not used as loss

        # copy and freeze the source model
        # if isinstance(model, ResNetDomainNet126):  # https://github.com/pytorch/pytorch/issues/28594
        #     for module in model.modules():
        #         for _, hook in module._forward_pre_hooks.items():
        #             if isinstance(hook, WeightNorm):
        #                 delattr(module, hook.name)

        # note: reduce memory consumption by only saving normalization parameters
        self.src_model = deepcopy(self.model).cpu()
        for param in self.src_model.parameters():
            param.detach_()
        self.mixed_precision=False
        # note: if the model is never reset, like for continual adaptation,
        # then skipping the state copy would save memory
        self.models = [self.src_model, self.model]
        # self.model_states, self.optimizer_state = self.copy_model_and_optimizer()

    def loss_calculation(self, x):
        imgs_test = x
        outputs = self.model(imgs_test)

        if self.use_weighting:
            with torch.no_grad():
                # calculate diversity based weight
                weights_div = 1 - F1.cosine_similarity(self.class_probs_ema.unsqueeze(dim=0), outputs.softmax(1), dim=1)
                weights_div = (weights_div - weights_div.min()) / (weights_div.max() - weights_div.min())
                mask = weights_div < weights_div.mean()

                # calculate certainty based weight
                weights_cert = - self.softmax_entropy(logits=outputs)
                weights_cert = (weights_cert - weights_cert.min()) / (weights_cert.max() - weights_cert.min())

                # calculate the final weights
                weights = torch.exp(weights_div * weights_cert / self.temperature)
                weights[mask] = 0.

                self.class_probs_ema = update_model_probs(x_ema=self.class_probs_ema, x=outputs.softmax(1).mean(0), momentum=self.momentum_probs)

        # calculate the soft likelihood ratio loss
        loss_out = self.slr(logits=outputs)

        # weight the loss
        if self.use_weighting:
            loss_out = loss_out * weights
            loss_out = loss_out[~mask]
        loss = loss_out.sum() / self.batch_size

        # calculate the consistency loss
        if self.use_consistency:
            outputs_aug = self.model(self.tta_transform(imgs_test[~mask]))
            loss += (self.symmetric_cross_entropy(x=outputs_aug, x_ema=outputs[~mask]) * weights[~mask]).sum() / self.batch_size

        return outputs, loss

    def forward(self,x):

        outputs = self.forward_and_adapt(x)
        return outputs

    @torch.enable_grad()
    def forward_and_adapt(self, x):
        if self.mixed_precision and self.device == "cuda":
            with torch.cuda.amp.autocast():
                outputs, loss = self.loss_calculation(x)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
        else:
            outputs, loss = self.loss_calculation(x)
            loss.backward()
            self.optimizer.step()
            self.optimizer.zero_grad()

        self.model = ema_update_model(
            model_to_update=self.model,
            model_to_merge=self.src_model,
            momentum=self.momentum_src,
            device=self.device
        )

        with torch.no_grad():
            if self.use_prior_correction:
                prior = outputs.softmax(1).mean(0)
                smooth = max(1 / outputs.shape[0], 1 / outputs.shape[1]) / torch.max(prior)
                smoothed_prior = (prior + smooth) / (1 + smooth * outputs.shape[1])
                outputs *= smoothed_prior

        return outputs

    def reset(self):
        if self.model_states is None or self.optimizer_state is None:
            raise Exception("cannot reset without saved model/optimizer state")
        self.load_model_and_optimizer()
        self.class_probs_ema = 1 / self.num_classes * torch.ones(self.num_classes).to(self.device)

def collect_params(model):
    """Collect the affine scale + shift parameters from normalization layers.
    Walk the model's modules and collect all normalization parameters.
    Return the parameters and their names.
    Note: other choices of parameterization are possible!
    """
    params = []
    names = []
    for nm, m in model.named_modules():
        if isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.GroupNorm)):
            for np, p in m.named_parameters():
                if np in ['weight', 'bias'] and p.requires_grad:
                    params.append(p)
                    names.append(f"{nm}.{np}")
    return params, names

def configure_model(model):

    model.eval()   # eval mode to avoid stochastic depth in swin. test-time normalization is still applied
    model.requires_grad_(False)  # disable grad, to (re-)enable only necessary parts
    # re-enable gradient for normalization layers
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.requires_grad_(True)
                # force use of batch stats in train and eval modes
            m.track_running_stats = False
            m.running_mean = None
            m.running_var = None
        elif isinstance(m, nn.BatchNorm1d):
            m.train()   # always forcing train mode in bn1d will cause problems for single sample tta
            m.requires_grad_(True)
        elif isinstance(m, (nn.LayerNorm, nn.GroupNorm)):
            m.requires_grad_(True)
    return model
