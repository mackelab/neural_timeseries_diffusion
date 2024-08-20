import torch

from ntd.modules import SLConv
from ntd.networks import CatConv, AdaConv


class TestSLConv:
    def randn_test(self, padding_mode):
        test_layer = SLConv(
            kernel_size=64,
            num_channels=100,
            num_scales=3,
            heads=3,
            padding_mode=padding_mode,
        )
        test_in = torch.randn(32, 100, 1000)

        with torch.no_grad():
            test_layer.use_fft_conv = True
            with_fft = test_layer.forward(test_in)

            test_layer.use_fft_conv = False
            wo_fft = test_layer.forward(test_in)

        assert torch.allclose(with_fft, wo_fft, atol=1e-5)

    def test_randn_zeros(self):
        self.randn_test("zeros")

    def test_randn_circular(self):
        self.randn_test("circular")


class TestRestrictedMode:
    def test_cat_conv(self, use_fft_conv=False):
        test = CatConv(
            signal_length=100,
            signal_channel=10,
            time_dim=16,
            cond_channel=3,
            hidden_channel=16,
            in_kernel_size=1,
            out_kernel_size=13,
            slconv_kernel_size=13,
            num_scales=3,
            heads=3,
            num_off_diag=0,
            use_fft_conv=use_fft_conv,
            use_pos_emb=True,
        )

        test_in_one = torch.randn(4, 10, 100)
        test_in_two = torch.randn(4, 10, 100)
        cond = torch.randn(4, 3, 100)
        one_channel = 3
        test_in_one[:, one_channel, :] = 1.0
        test_in_two[:, one_channel, :] = 1.0
        arange_channel = 6
        test_in_one[:, arange_channel, :] = torch.arange(100)
        test_in_two[:, arange_channel, :] = torch.arange(100)

        res_one = test.forward(test_in_one, torch.tensor([0, 1, 2, 3]), cond=cond)
        res_two = test.forward(test_in_two, torch.tensor([0, 1, 2, 3]), cond=cond)

        assert torch.allclose(res_one[:, one_channel, :], res_two[:, one_channel, :])
        assert torch.allclose(
            res_one[:, arange_channel, :], res_two[:, arange_channel, :]
        )
        assert not torch.allclose(res_one, res_two)

    def test_cat_conv_fft(self):
        self.test_cat_conv(use_fft_conv=True)

    def test_ada_conv(self, use_fft_conv=False):
        test = AdaConv(
            signal_length=100,
            signal_channel=10,
            cond_dim=3,
            hidden_channel=16,
            in_kernel_size=1,
            out_kernel_size=13,
            slconv_kernel_size=13,
            num_scales=3,
            num_blocks=3,
            num_off_diag=0,
            use_fft_conv=use_fft_conv,
            use_pos_emb=True,
        )

        test_in_one = torch.randn(4, 10, 100)
        test_in_two = torch.randn(4, 10, 100)
        cond = torch.randn(4, 3, 100)
        one_channel = 3
        test_in_one[:, one_channel, :] = 1.0
        test_in_two[:, one_channel, :] = 1.0
        arange_channel = 6
        test_in_one[:, arange_channel, :] = torch.arange(100)
        test_in_two[:, arange_channel, :] = torch.arange(100)

        res_one = test.forward(test_in_one, torch.tensor([0, 1, 2, 3]), cond=cond)
        res_two = test.forward(test_in_two, torch.tensor([0, 1, 2, 3]), cond=cond)

        assert torch.allclose(res_one[:, one_channel, :], res_two[:, one_channel, :])
        assert torch.allclose(
            res_one[:, arange_channel, :], res_two[:, arange_channel, :]
        )
        assert not torch.allclose(res_one, res_two)

    def test_ada_conv_fft(self):
        self.test_ada_conv(use_fft_conv=True)
