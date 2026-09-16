import pytest
import os
import sys
from typing import List, Dict, Any
from unittest.mock import MagicMock, patch, ANY

from beni.core.srl import (
    MasterConfig, ModelConfig, DataConfig, GRPOTrainerConfig,
    DistillationConfig, RewardConfig, SRLGrpoPrompt, SRLTrainer
)
from beni.core.compute.rewards import RewardManager
from beni.core.srl.grpo.grpo import SebeniGrpo

TEST_MODELS = [
    "HuggingFaceTB/SmolLM2-135M",
    "facebook/MobileLLM-350M",
]


class TestGRPOConfigs:
    """Test suite for GRPO pipeline configurations."""

    def test_default_master_config(self):
        config = MasterConfig()
        assert config.project_name == "GRPO-Morphology-Advanced"
        assert config.model.model_name == "HuggingFaceTB/SmolLM2-135M"
        assert config.distillation.enabled is True
        assert config.distillation.provider == "google"
        assert config.reward.format_weight == 0.2
        assert config.reward.morph_weight == 0.4
        assert config.reward.rule_weight == 0.4

    def test_master_config_from_dict(self):
        config_dict = {
            "project_name": "Custom-Morph-Experiment",
            "model": {"model_name": "facebook/MobileLLM-350M", "load_in_4bit": True},
            "distillation": {"enabled": False, "provider": "gemini", "model": "gemini-2.5-flash"},
            "reward": {"format_weight": 0.5, "morph_weight": 0.5, "enable_rule_reward": False}
        }
        config = MasterConfig.from_dict(config_dict)
        assert config.project_name == "Custom-Morph-Experiment"
        assert config.model.model_name == "facebook/MobileLLM-350M"
        assert config.distillation.enabled is False
        assert config.distillation.provider == "gemini"
        assert config.reward.format_weight == 0.5
        assert config.reward.enable_rule_reward is False

    def test_srl_trainer_is_facade(self):
        from beni.core.srl.unified import SRLTrainer
        from beni.core.srl.grpo.grpo import SebeniGrpo
        assert SRLTrainer is not SebeniGrpo
        trainer = SRLTrainer()
        assert isinstance(trainer.plugin, SebeniGrpo)


class TestRewardManager:
    """Test suite for RewardManager and custom reward registration."""

    def test_default_reward_functions(self):
        rm = RewardManager()
        funcs = rm.get_reward_functions()
        assert len(funcs) == 4
        func_names = [f.__name__ for f in funcs]
        assert "reward_format" in func_names
        assert "reward_morph" in func_names
        assert "reward_rule" in func_names
        assert "reward_lang" in func_names

    def test_disabled_reward_functions(self):
        cfg = RewardConfig(enable_rule_reward=False)
        rm = RewardManager(reward_config=cfg)
        funcs = rm.get_reward_functions()
        assert len(funcs) == 3
        func_names = [f.__name__ for f in funcs]
        assert "reward_rule" not in func_names

    def test_custom_reward_registration(self):
        rm = RewardManager()

        def custom_length_reward(completions: List[str], **kwargs) -> List[float]:
            return [float(len(c)) for c in completions]

        rm.register_custom_reward("length_reward", custom_length_reward, weight=0.5)
        funcs = rm.get_reward_functions()
        assert len(funcs) == 5

        # Test execution of custom reward wrapper
        last_func = funcs[-1]
        results = last_func(["hello", "world!!"])
        assert results == [2.5, 3.5]  # 5 * 0.5 and 7 * 0.5


class TestSebeniGrpoPipeline:
    """Test suite for SebeniGrpo pipeline functionality."""

    def test_pipeline_initialization(self):
        pipeline = SebeniGrpo()
        assert pipeline.config is not None
        assert pipeline.reward_manager is not None
        assert pipeline.hook_manager is not None
        assert pipeline.config.model.model_name == "HuggingFaceTB/SmolLM2-135M"

    def test_custom_config_initialization(self):
        master_cfg = MasterConfig(project_name="Pipeline-Test")
        pipeline = SebeniGrpo(config=master_cfg)
        assert pipeline.config.project_name == "Pipeline-Test"

    def test_register_reward_and_hook(self):
        pipeline = SebeniGrpo()

        def dummy_reward(completions, **kwargs):
            return [1.0] * len(completions)

        def dummy_hook(model, ref_model, batch_meta):
            pass

        pipeline.register_reward("dummy", dummy_reward, weight=0.2)
        pipeline.register_hook(dummy_hook)

        assert "dummy" in pipeline.reward_manager.custom_rewards
        assert len(pipeline.hook_manager.hooks) == 1

    def test_format_dataset(self):
        pipeline = SebeniGrpo()
        raw_sentences = [
            {
                "text": "aw ka ne labato.",
                "lang": "bm",
                "reference": {"tokens": [{"surface": "aw"}]}
            },
            {
                "text": "kàlanko jamanaw",
                "lang": "bm",
                "reference": {"tokens": [{"surface": "kàlanko"}]}
            }
        ]

        ds = pipeline.format_dataset(raw_sentences)

        assert len(ds) == 2
        assert "prompt" in ds.column_names
        assert "language" in ds.column_names
        assert "reference" in ds.column_names

        first_prompt = ds[0]["prompt"]
        assert first_prompt[0]["role"] == "system"
        assert first_prompt[1]["role"] == "user"
        assert first_prompt[1]["content"] == "aw ka ne labato."
        assert ds[0]["language"] == "bm"

    def test_run_batch_distillation_disabled(self):
        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = False
        raw_sentences = [
            {"text": "test sentence", "lang": "bm"}
        ]
        results = pipeline.run_batch_distillation(raw_sentences)
        assert results == {}

    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_run_batch_distillation_enabled(self, mock_distiller_cls):
        mock_distiller_instance = MagicMock()
        mock_distiller_instance.run_batch_distillation.return_value = (
            "baseline_v2.gram", "baseline_v2.dict"
        )
        mock_distiller_cls.return_value = mock_distiller_instance

        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = True
        pipeline.config.distillation.provider = "google"

        raw_sentences = [
            {"text": "aw ka ne labato.", "lang": "bm"},
            {"text": "kàlanko jamanaw", "lang": "bm"}
        ]

        results = pipeline.run_batch_distillation(raw_sentences)

        mock_distiller_cls.assert_called_once_with(
            lang_code="bam",
            provider="google",
            model="gemini-2.5-flash",
            working_dir=None
        )
        mock_distiller_instance.run_batch_distillation.assert_called_once_with([
            "aw ka ne labato.", "kàlanko jamanaw"
        ])
        assert "bam" in results
        assert results["bam"]["gram_path"] == "baseline_v2.gram"
        assert results["bam"]["dict_path"] == "baseline_v2.dict"

    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_run_batch_distillation_mixed_languages(self, mock_distiller_cls):
        mock_distiller_instance = MagicMock()
        mock_distiller_instance.run_batch_distillation.return_value = ("g.gram", "d.dict")
        mock_distiller_cls.return_value = mock_distiller_instance

        pipeline = SebeniGrpo()
        pipeline.config.distillation.enabled = True
        results = pipeline.run_batch_distillation([
            {"text": "aw ka", "lang": "bam"},
            {"text": "i ni ce", "lang": "mku"},
        ])
        assert mock_distiller_cls.call_count == 2
        codes = {c.kwargs["lang_code"] for c in mock_distiller_cls.call_args_list}
        assert codes == {"bam", "mku"}
        assert set(results) == {"bam", "mku"}


class TestGRPOTrainingModel:
    """Test suite for model loading, PEFT adaptation, and training loop execution."""

    @pytest.mark.parametrize("model_name", TEST_MODELS)
    @patch("beni.core.srl.plugin.get_peft_model")
    @patch("beni.core.srl.plugin.AutoTokenizer.from_pretrained")
    @patch("beni.core.srl.plugin.AutoModelForCausalLM.from_pretrained")
    def test_load_models_and_lora(self, mock_model_from_pretrained, mock_tok_from_pretrained, mock_get_peft, model_name):
        mock_base_model = MagicMock()
        mock_peft_model = MagicMock()
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"

        mock_model_from_pretrained.return_value = mock_base_model
        mock_tok_from_pretrained.return_value = mock_tokenizer
        mock_get_peft.return_value = mock_peft_model

        pipeline = SebeniGrpo(model_name=model_name)
        pipeline.config.model.use_peft = True
        pipeline.config.model.load_in_4bit = True

        pipeline.load_models()

        mock_model_from_pretrained.assert_called_once()
        assert mock_tok_from_pretrained.call_args[0][0] == model_name
        mock_get_peft.assert_called_once_with(mock_base_model, ANY)
        assert pipeline.model == mock_peft_model
        assert pipeline.tokenizer == mock_tokenizer

    @patch("beni.core.srl.plugin.write_model_card")
    @patch("beni.core.srl.grpo.grpo.GRPOConfig")
    @patch("beni.core.srl.grpo.grpo.GRPOTrainer")
    @patch("beni.core.srl.grpo.grpo.Distiller")
    def test_train_pipeline_execution(self, mock_distiller_cls, mock_trainer_cls, mock_grpo_config, mock_card, tmp_path):
        mock_distiller_inst = MagicMock()
        mock_distiller_inst.run_batch_distillation.return_value = ("v1.gram", "v1.dict")
        mock_distiller_cls.return_value = mock_distiller_inst

        mock_trainer_inst = MagicMock()
        mock_trainer_inst.train.return_value = {"train_loss": 0.123}
        mock_trainer_inst.optimizer = None
        mock_trainer_cls.return_value = mock_trainer_inst
        mock_grpo_config.return_value = MagicMock()

        pipeline = SebeniGrpo(model_name="HuggingFaceTB/SmolLM2-135M")
        pipeline.model = MagicMock()  # pre-loaded mock model
        pipeline.tokenizer = MagicMock()
        pipeline.config.distillation.enabled = True
        pipeline.config.trainer.use_cpu = True
        pipeline.config.trainer.output_dir = str(tmp_path)
        pipeline.config.trainer.push_to_hub = False

        raw_sentences = [
            {"text": "aw ka ne labato.", "lang": "bm", "reference": {"tokens": []}}
        ]

        result = pipeline.train(raw_sentences, project_name="Test-Run")

        mock_distiller_inst.run_batch_distillation.assert_called_once()
        mock_trainer_cls.assert_called_once()
        mock_trainer_inst.train.assert_called_once()
        assert result == {"train_loss": 0.123}
