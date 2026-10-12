"""Strict experiment configuration shared by all architectures."""
from dataclasses import dataclass,asdict,field
from kssd.models import ARCHITECTURES


@dataclass
class ExperimentConfig:
    architecture: str='modified_unet'
    epochs: int=10
    batch_size: int=2
    precision: str='float32'
    learning_rate: float=.001
    optimizer: str='adam'
    loss: str='dice'
    threshold: float=.5
    seed: int=42
    n_folds: int=5
    mask_threshold: int=127
    input_size: int=512
    early_stopping_patience: int=0
    augmentation: bool=False
    augmentation_parameters: dict=field(default_factory=dict)

    def __post_init__(self):
        if self.precision not in ['float32','mixed_float16']:raise ValueError('Unsupported precision')
        if self.architecture not in ARCHITECTURES:raise ValueError('Unknown architecture')
        if self.epochs<1 or self.batch_size<1 or self.learning_rate<=0:raise ValueError('Invalid training budget')
        if self.optimizer not in ['adam','sgd']:raise ValueError('Unsupported optimizer')
        if self.loss not in ['dice','focal_tversky']:raise ValueError('Unsupported loss')
        if not 0<=self.threshold<=1 or not 0<=self.mask_threshold<=254:raise ValueError('Invalid threshold')
        if self.input_size<32 or self.input_size%16:raise ValueError('Invalid input size')
        if self.n_folds<2 or self.early_stopping_patience<0:raise ValueError('Invalid fold/stopping configuration')

    def to_dict(self):return asdict(self)
