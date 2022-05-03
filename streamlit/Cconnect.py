import torch 
import os
import numpy as np
from PIL import Image
from trainedmodel import Model

model = Model(*args, **kwargs)
PATH = "ptModel/model.pt"
device = torch.device('cpu')
model.load_state_dict(torch.load(PATH))
model.eval()