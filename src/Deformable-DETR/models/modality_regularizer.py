import torch
import random
from torch import nn

FIXED_P, VAR_P_BATCH, VAR_P_IMG, LEARNABLE_P_IMG, LEARNABLE_P_BATCH, CURRICULUM_P = 'fixed', 'variable_per_batch', 'variable_per_img', 'learnable_per_img', 'learnable_per_batch', 'curriculum_per_patch'
NONE, SINGLE_MODALITY, MOOD_MODALITY, SINGLE_OR_MOOD = None, 'single_modality', 'mood_modality', 'single_or_mood'

def equals(str1, str2): return str1.casefold() == str2.casefold()

class ModalityRegularizer(nn.Module):
    def __init__(self, strong_modality_no: int = 0, regularization_method: str = NONE, prob: int = 5):
        super().__init__()
        self.method = regularization_method if regularization_method is not None else ''
        self.strong_modality_no = strong_modality_no
        self.prob = prob
    
    def single_modality(self, mod1, mod2):
        mods = [mod1, mod2]
        mods[self.strong_modality_no] = torch.zeros(mod1.shape, device=mod1.device)

        return mods

    def mood_modality(self, mod1, mod2):
        mods = [mod1, mod2]
        mods[self.strong_modality_no] = (mod1 + mod2) / 2

        return mods

    def forward(self, mod1, mod2):
        with torch.no_grad():
            dice = random.randint(0, self.prob)
            if dice == 0 or dice == 2:
                if equals(self.method, SINGLE_MODALITY):
                    mod1, mod2 = self.single_modality(mod1, mod2)
                elif equals(self.method, MOOD_MODALITY):
                    mod1, mod2 = self.mood_modality(mod1, mod2)
                elif equals(self.method, SINGLE_OR_MOOD):
                    mod1, mod2 = self.mood_modality(mod1, mod2) if bool(random.getrandbits(1)) else self.single_modality(mod1, mod2)
            
            return mod1, mod2