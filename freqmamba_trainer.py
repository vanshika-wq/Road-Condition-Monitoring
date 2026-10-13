
import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from freqmamba_rdd.freqmamba_rdd import FreqMambaRDD

class FreqMambaAdapter(nn.Module):
    def __init__(self, channels=128):
        super().__init__()
        self.reduce = nn.Conv2d(channels, 32, 1)
        self.freqmamba = FreqMambaRDD(32)
        self.expand = nn.Conv2d(32, channels, 1)

    def forward(self, x):
        return x + self.expand(self.freqmamba(self.reduce(x)))

class EnhancedLayer(nn.Module):
    def __init__(self, original, adapter):
        super().__init__()
        self.original = original
        self.adapter = adapter
        for attr in ("f", "i", "type", "np"):
            if hasattr(original, attr):
                setattr(self, attr, getattr(original, attr))

    def forward(self, x):
        return self.adapter(self.original(x))

class FreqMambaTrainer(DetectionTrainer):
    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        model.model[4] = EnhancedLayer(model.model[4], FreqMambaAdapter(128))
        return model
