import importlib
import unittest

import torch

_r2_model = importlib.import_module(
    "train.0014_faithful_board_causal_features.r2_model"
)
_r4_model = importlib.import_module(
    "train.0014_faithful_board_causal_features.r4_model"
)
ScenarioScaleGate = _r2_model.ScenarioScaleGate
OptionFamilyRoute = _r4_model.OptionFamilyRoute


class ScenarioScaleGateTests(unittest.TestCase):
    def test_initial_scale_is_one_and_stays_in_open_zero_two_range(self) -> None:
        gate = ScenarioScaleGate(input_width=12, d_model=4, multiplier=2)
        values = gate(torch.randn(5, 12))
        torch.testing.assert_close(values, torch.ones_like(values))

        output = gate.network[-1]
        self.assertIsInstance(output, torch.nn.Linear)
        with torch.no_grad():
            output.weight.normal_(std=0.2)
            output.bias.normal_(std=0.2)
        values = gate(torch.randn(5, 12))
        self.assertTrue(bool((values > 0).all()))
        self.assertTrue(bool((values < 2).all()))

    def test_r4_family_route_honors_declared_initial_scale(self) -> None:
        route = OptionFamilyRoute(8, 2, initial_scale=0.25)
        evidence = torch.randn(3, 5, 8)
        scale = 2.0 * torch.sigmoid(route.gate(evidence))
        torch.testing.assert_close(scale, torch.full_like(scale, 0.25))


if __name__ == "__main__":
    unittest.main()
