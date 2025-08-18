from dataclasses import dataclass

@dataclass
class Config:
    ticker: str = "GS"
    start: str = "2010-01-01"
    end: str = "2018-12-31"
    test_years: int = 2

    seq_len: int = 17
    batch_size: int = 64
    lr_g: float = 1e-3
    lr_d: float = 1e-4
    n_epochs: int = 20
    critic_steps: int = 5
    hidden_size: int = 64
    num_layers: int = 1

    fourier_k: int = 10
    arima_order: tuple = (5, 1, 0)

    ae_hidden: int = 64
    ae_bottleneck: int = 32
    ae_epochs: int = 10

    pca_components: int = 12

    use_amp: bool = True
