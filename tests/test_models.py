"""Tests for neural network models (generators, discriminator, GAN)."""
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from src.models.generator import LSTMGenerator, TransformerGenerator
from src.models.discriminator import CNNDiscriminator
from src.models.gan import WGAN_GP, gradient_penalty


class TestLSTMGenerator:
    def test_output_shapes(self):
        B, T, F = 8, 10, 20
        gen = LSTMGenerator(input_size=F, hidden_size=32, num_layers=1,
                            num_quantiles=3, dropout=0.1)
        x = torch.randn(B, T, F)
        y_reg, y_cls, y_q = gen(x)
        assert y_reg.shape == (B,)
        assert y_cls.shape == (B,)
        assert y_q.shape == (B, 3)

    def test_quantile_head_count(self):
        gen = LSTMGenerator(input_size=10, num_quantiles=5, dropout=0.0)
        x = torch.randn(4, 8, 10)
        _, _, y_q = gen(x)
        assert y_q.shape == (4, 5)

    def test_dropout_param(self):
        gen = LSTMGenerator(input_size=10, dropout=0.5)
        assert gen.drop.p == 0.5


class TestTransformerGenerator:
    def test_output_shapes(self):
        B, T, F = 8, 10, 20
        gen = TransformerGenerator(input_size=F, d_model=32, nhead=2,
                                   num_layers=1, num_quantiles=3)
        x = torch.randn(B, T, F)
        y_reg, y_cls, y_q = gen(x)
        assert y_reg.shape == (B,)
        assert y_cls.shape == (B,)
        assert y_q.shape == (B, 3)

    def test_different_seq_lens(self):
        gen = TransformerGenerator(input_size=10, d_model=32, nhead=2,
                                   num_layers=1, num_quantiles=3)
        for T in [5, 10, 30]:
            x = torch.randn(4, T, 10)
            y_reg, y_cls, y_q = gen(x)
            assert y_reg.shape == (4,)


class TestCNNDiscriminator:
    def test_output_shape(self):
        B, T, F = 8, 10, 20
        disc = CNNDiscriminator(input_size=F, seq_len=T)
        x = torch.randn(B, T, F)
        y = torch.randn(B)
        score = disc(x, y)
        assert score.shape == (B,)

    def test_discriminator_uses_y(self):
        """Verify that different y values produce different scores."""
        B, T, F = 8, 10, 20
        disc = CNNDiscriminator(input_size=F, seq_len=T)
        x = torch.randn(B, T, F)
        y1 = torch.ones(B) * 10.0
        y2 = torch.ones(B) * -10.0
        s1 = disc(x, y1)
        s2 = disc(x, y2)
        assert not torch.allclose(s1, s2, atol=1e-6), \
            "Discriminator output is the same for different y -- it ignores y!"

    def test_has_batchnorm(self):
        """Verify discriminator includes BatchNorm1d layers."""
        disc = CNNDiscriminator(input_size=10, seq_len=8)
        bn_count = sum(1 for m in disc.modules()
                       if isinstance(m, torch.nn.BatchNorm1d))
        assert bn_count >= 3, f"Expected >=3 BatchNorm1d layers, found {bn_count}"


class TestGradientPenalty:
    def test_finite(self):
        B, T, F = 8, 10, 20
        disc = CNNDiscriminator(input_size=F, seq_len=T)
        x = torch.randn(B, T, F)
        real_y = torch.randn(B)
        fake_y = torch.randn(B)
        gp = gradient_penalty(disc, x, real_y, fake_y, "cpu")
        assert torch.isfinite(gp).item(), "Gradient penalty should be finite"
        assert gp.item() >= 0, "Gradient penalty should be non-negative"


class TestWGANGP:
    def _make_loader(self, B=16, T=10, F=20):
        X = torch.randn(B, T, F)
        y = torch.randn(B)
        ds = TensorDataset(X, y)
        return DataLoader(ds, batch_size=8, shuffle=True), F, T

    def test_train_epoch_lstm(self):
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1,
                      num_quantiles=3, dropout_lstm=0.1,
                      use_lr_scheduler=False)
        g_loss, d_loss = gan.train_epoch(dl)
        assert np.isfinite(g_loss)
        assert np.isfinite(d_loss)

    def test_train_epoch_tst(self):
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="tst", d_model=32, nhead=2,
                      num_layers_tst=1, critic_steps=1, num_quantiles=3,
                      use_lr_scheduler=False)
        g_loss, d_loss = gan.train_epoch(dl)
        assert np.isfinite(g_loss)
        assert np.isfinite(d_loss)

    def test_predict_shapes(self):
        dl, F, T = self._make_loader(B=8)
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1, num_quantiles=3,
                      use_lr_scheduler=False)
        xb = torch.randn(4, T, F)
        y_reg, y_logit, y_q = gan.predict(xb)
        assert y_reg.shape == (4,)
        assert y_logit.shape == (4,)
        assert y_q.shape == (4, 3)

    def test_quantile_output_shapes(self):
        dl, F, T = self._make_loader(B=8)
        for gen_type in ["lstm", "tst"]:
            gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                          generator_type=gen_type, d_model=32, nhead=2,
                          num_layers_tst=1, critic_steps=1,
                          num_quantiles=5, quantiles=(0.1, 0.25, 0.5, 0.75, 0.9),
                          use_lr_scheduler=False)
            xb = torch.randn(4, T, F)
            _, _, y_q = gan.predict(xb)
            assert y_q.shape == (4, 5), f"Quantile shape wrong for {gen_type}"

    def test_lr_scheduler_steps(self):
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1, num_quantiles=3,
                      use_lr_scheduler=True, n_epochs=10)
        assert gan.sched_g is not None
        assert gan.sched_d is not None
        initial_lr = gan.opt_g.param_groups[0]["lr"]
        gan.train_epoch(dl)
        gan.step_schedulers()
        new_lr = gan.opt_g.param_groups[0]["lr"]
        # After one step of CosineAnnealing, LR should have decreased
        assert new_lr <= initial_lr

    def test_no_scheduler_when_disabled(self):
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1,
                      use_lr_scheduler=False)
        assert gan.sched_g is None
        assert gan.sched_d is None
        # step_schedulers should not crash
        gan.step_schedulers()

    def test_grad_clip_zero_disables(self):
        """grad_clip=0 should disable gradient clipping (no error)."""
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1,
                      grad_clip=0.0, use_lr_scheduler=False)
        g_loss, d_loss = gan.train_epoch(dl)
        assert np.isfinite(g_loss)
        assert np.isfinite(d_loss)

    def test_grad_clip_nonzero(self):
        """grad_clip > 0 should clip gradients (no error, finite losses)."""
        dl, F, T = self._make_loader()
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=1,
                      grad_clip=0.5, use_lr_scheduler=False)
        g_loss, d_loss = gan.train_epoch(dl)
        assert np.isfinite(g_loss)
        assert np.isfinite(d_loss)

    def test_multiple_epochs_losses_change(self):
        """Losses should generally change across epochs (model is learning)."""
        dl, F, T = self._make_loader(B=32)
        gan = WGAN_GP(input_size=F, seq_len=T, hidden_size=32,
                      generator_type="lstm", critic_steps=2,
                      use_lr_scheduler=False)
        losses = []
        for _ in range(3):
            g, d = gan.train_epoch(dl)
            losses.append((g, d))
        # At least one loss should differ between epochs
        g_losses = [l[0] for l in losses]
        assert len(set(round(g, 4) for g in g_losses)) > 1


class TestPinballLossTorch:
    def test_zero_at_perfect(self):
        from src.models.gan import pinball_loss_torch
        y_true = torch.tensor([1.0, 2.0, 3.0])
        y_q = torch.column_stack([y_true, y_true, y_true])
        loss = pinball_loss_torch(y_true, y_q, (0.1, 0.5, 0.9))
        assert loss.item() < 1e-6

    def test_positive_on_errors(self):
        from src.models.gan import pinball_loss_torch
        y_true = torch.tensor([0.0, 0.0, 0.0])
        y_q = torch.tensor([[1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0],
                             [1.0, 1.0, 1.0]])
        loss = pinball_loss_torch(y_true, y_q, (0.1, 0.5, 0.9))
        assert loss.item() > 0

    def test_gradient_flows(self):
        from src.models.gan import pinball_loss_torch
        y_true = torch.tensor([1.0, 2.0])
        y_q = torch.tensor([[0.5, 1.0, 1.5],
                             [1.5, 2.0, 2.5]], requires_grad=True)
        loss = pinball_loss_torch(y_true, y_q, (0.1, 0.5, 0.9))
        loss.backward()
        assert y_q.grad is not None
        assert torch.isfinite(y_q.grad).all()

    def test_single_quantile(self):
        from src.models.gan import pinball_loss_torch
        y_true = torch.tensor([1.0, 2.0])
        y_q = torch.tensor([[1.5], [2.5]])
        loss = pinball_loss_torch(y_true, y_q, (0.5,))
        assert torch.isfinite(loss)


class TestPositionalEncoding:
    def test_output_shape(self):
        from src.models.generator import PositionalEncoding
        pe = PositionalEncoding(d_model=32, max_len=100)
        x = torch.randn(4, 20, 32)
        out = pe(x)
        assert out.shape == (4, 20, 32)

    def test_odd_d_model(self):
        from src.models.generator import PositionalEncoding
        pe = PositionalEncoding(d_model=33, max_len=100)
        x = torch.randn(2, 10, 33)
        out = pe(x)
        assert out.shape == (2, 10, 33)
        assert torch.isfinite(out).all()

    def test_adds_positional_info(self):
        from src.models.generator import PositionalEncoding
        pe = PositionalEncoding(d_model=16, max_len=50, dropout=0.0)
        x = torch.zeros(1, 10, 16)
        out = pe(x)
        # With zero input and no dropout, output should be the PE buffer itself
        assert not torch.allclose(out, x), "PE should add non-zero positional info"


class TestLSTMGeneratorMultiLayer:
    def test_multi_layer_output_shapes(self):
        gen = LSTMGenerator(input_size=10, hidden_size=32, num_layers=3,
                            num_quantiles=3, dropout=0.2)
        x = torch.randn(4, 8, 10)
        y_reg, y_cls, y_q = gen(x)
        assert y_reg.shape == (4,)
        assert y_cls.shape == (4,)
        assert y_q.shape == (4, 3)

    def test_multi_layer_trains(self):
        gen = LSTMGenerator(input_size=10, hidden_size=32, num_layers=2,
                            num_quantiles=3, dropout=0.1)
        x = torch.randn(8, 5, 10)
        y_reg, _, _ = gen(x)
        loss = y_reg.sum()
        loss.backward()
        # Check at least some gradients flow (output layers always get grads)
        grads_found = sum(1 for p in gen.parameters()
                          if p.requires_grad and p.grad is not None)
        assert grads_found > 0, "No gradients found in any parameter"


class TestCNNDiscriminatorEdgeCases:
    def test_various_seq_lens(self):
        for T in [5, 10, 20, 50]:
            F = 15
            disc = CNNDiscriminator(input_size=F, seq_len=T)
            x = torch.randn(4, T, F)
            y = torch.randn(4)
            score = disc(x, y)
            assert score.shape == (4,), f"Failed for seq_len={T}"

    def test_single_sample(self):
        disc = CNNDiscriminator(input_size=10, seq_len=8)
        x = torch.randn(1, 8, 10)
        y = torch.randn(1)
        score = disc(x, y)
        assert score.shape == (1,)
