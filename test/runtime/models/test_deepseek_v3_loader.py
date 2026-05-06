import unittest
from unittest import mock

from tokenspeed.runtime.models.deepseek_v3 import DeepseekV3ForCausalLM


class TestDeepseekV3Loader(unittest.TestCase):
    def test_missing_checkpoint_scale_params_are_silent(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            self.assertIsNone(
                model.get_param(
                    {},
                    "model.layers.2.self_attn.k_proj.k_scale",
                )
            )
            self.assertIsNone(
                model.get_param(
                    {},
                    "model.layers.2.self_attn.v_proj.v_scale",
                )
            )

        logger.warning.assert_not_called()

    def test_missing_regular_params_still_warn(self):
        model = object.__new__(DeepseekV3ForCausalLM)

        with mock.patch("tokenspeed.runtime.models.deepseek_v3.logger") as logger:
            self.assertIsNone(
                model.get_param({}, "model.layers.2.self_attn.q_proj.weight")
            )

        logger.warning.assert_called_once()


if __name__ == "__main__":
    unittest.main()
