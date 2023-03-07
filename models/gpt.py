from typing import List
from torch import nn
import torch
from pathlib import Path
import json
from gpt_model import Model, HParams

class GPTModel(nn.Module):
    def __init__(
        self,
        path
    ):
        super().__init__()
        root = Path(path)

        params = json.loads((root / 'params.json').read_text())
        hparams = params['hparams']
        hparams.setdefault('n_hidden', hparams['n_embed'])
        self.model = Model(HParams(**hparams))
        state = torch.load(root / 'model.pt', map_location='cpu')
        state_dict = self.fixed_state_dict(state['state_dict'])
        self.model.load_state_dict(state_dict)
        for param in self.model.parameters():
          param.requires_grad = False
        self.fc = nn.Linear(500, 17)

    def fixed_state_dict(self, state_dict):
      if all(k.startswith('module.') for k in state_dict):
          # legacy multi-GPU format
          state_dict = {k[len('module.'):]: v for k, v in state_dict.items()}
      return state_dict

    def forward(self, src: torch.Tensor, lengths: torch.Tensor, target=None):

        # logits shape [batch_size, 256, 500]
        logits = self.model(src)['logits']
        predictions = self.fc(logits)
      
        output = {"diacritics": predictions}

        return output
