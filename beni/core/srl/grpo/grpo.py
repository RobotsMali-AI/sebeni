from typing import Any, Dict, List, Optional, Union

from datasets import Dataset
from trl import GRPOConfig, GRPOTrainer

from beni.core.srl.config import (
    MasterConfig, ModelConfig, DataConfig, GRPOTrainerConfig,
    DistillationConfig, RewardConfig
)
from beni.core.srl.plugin import AlignmentPlugin
from beni.core.morphotactic.distil.distillation import Distiller
from beni.core.srl.grpo.callbacks import TrackioMetricsCallback, SelfAwareCallback
from beni.core.srl.algorithm1 import confirm_hitl


class SebeniGrpo(AlignmentPlugin):
    """
    GRPO policy-update plugin for SAMPG.

    Distiller is invoked per batch when Φ < τ via SelfAwareCallback.
    An optional initial distill still runs when no baseline exists.
    """

    name = "grpo"

    def __init__(
        self,
        config: Optional[MasterConfig] = None,
        model_name: Optional[str] = None,
        data_config: Optional[DataConfig] = None,
        model_config: Optional[ModelConfig] = None,
        trainer_config: Optional[GRPOTrainerConfig] = None,
        distillation_config: Optional[DistillationConfig] = None,
        reward_config: Optional[RewardConfig] = None
    ):
        super().__init__(
            config=config,
            model_name=model_name,
            data_config=data_config,
            model_config=model_config,
            trainer_config=trainer_config,
            distillation_config=distillation_config,
            reward_config=reward_config,
        )

    def run_batch_distillation(self, sentences: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Initial / HITL Distiller pass (not the per-batch Φ < τ path)."""
        if not self.config.distillation.enabled:
            print("Morphotactic distillation is disabled in config.")
            return {}

        from beni.core.srl.algorithm1 import group_texts_by_language, records_text_langs

        default_lang = self.config.data.default_lang or "bam"
        lang_groups = group_texts_by_language(
            records_text_langs(sentences, default_lang),
            default_lang=default_lang,
        )
        allowed = set(self.config.languages()) if self.config.data.languages else None

        distillation_results = {}
        for group_code, texts in lang_groups.items():
            if allowed is not None and group_code not in allowed:
                continue
            print(f"Running Morphotactic Distillation on batch for language '{group_code}' ({len(texts)} texts)...")
            try:
                distiller = Distiller(
                    lang_code=group_code,
                    provider=self.config.distillation.provider,
                    model=self.config.distillation.model,
                    working_dir=self.config.distillation.working_dir
                )
                if self.config.distillation.hitl:
                    distiller.handle_baselines()
                    proposal = distiller.propose(texts)
                    if proposal is None:
                        distillation_results[group_code] = {"error": "no proposal"}
                        continue
                    if not confirm_hitl(proposal.gram_text, proposal.dict_text, True):
                        distillation_results[group_code] = {"skipped": "hitl_rejected"}
                        continue
                    decision = self.governor.allow_promote(
                        proposal.phi,
                        proposal.phi_prime,
                        parseable=proposal.parseable,
                        first_create=proposal.first_create,
                        language_ok=proposal.language_ok,
                    )
                    if decision.allowed:
                        distiller.write_checkpoint(proposal.gram_text, proposal.dict_text)
                        distillation_results[group_code] = {
                            "gram_path": str(distiller.gram_path),
                            "dict_path": str(distiller.dict_path),
                            "reason": decision.reason,
                        }
                    else:
                        distillation_results[group_code] = {"error": decision.reason}
                    continue

                result = distiller.run_batch_distillation(texts)
                if result and result[0] and result[1]:
                    gram_path, dict_path = result
                    distillation_results[group_code] = {
                        "gram_path": gram_path,
                        "dict_path": dict_path
                    }
                    print(f"Distillation complete for '{group_code}': gram={gram_path}, dict={dict_path}")
                else:
                    distillation_results[group_code] = {"error": "no baseline update"}
            except Exception as e:
                print(f"Morphotactic distillation for '{group_code}' skipped: {e}")
                distillation_results[group_code] = {"error": str(e)}

        return distillation_results

    def train(
        self,
        data: Union[List[Dict[str, Any]], Dataset],
        project_name: Optional[str] = None,
        run_distillation_first: bool = True,
        extra_callbacks: Optional[List] = None,
    ):
        """Execute GRPO with SelfAwareCallback on each batch."""
        raw_sentences = None
        if isinstance(data, list):
            raw_sentences = data
            if run_distillation_first and self.config.distillation.enabled:
                self.run_batch_distillation(raw_sentences)
            dataset = self.format_dataset(raw_sentences)
        else:
            dataset = data

        if self.model is None:
            self.load_models()

        project_name = project_name or self.config.project_name
        grpo_args = GRPOConfig(**self.config.trainer.to_dict())
        reward_funcs = self.reward_manager.get_reward_functions()

        callbacks = [TrackioMetricsCallback(reward_manager=self.reward_manager)]
        if extra_callbacks:
            callbacks.extend(extra_callbacks)
        if self.config.distillation.enabled and not any(
            isinstance(cb, SelfAwareCallback) for cb in callbacks
        ):
            callbacks.append(SelfAwareCallback(self.config, governor=self.governor))

        trainer_kwargs = {
            "model": self.model,
            "reward_funcs": reward_funcs,
            "args": grpo_args,
            "train_dataset": dataset,
            "callbacks": callbacks,
        }
        if self.tokenizer is not None:
            trainer_kwargs["processing_class"] = self.tokenizer

        self.trainer = GRPOTrainer(**trainer_kwargs)
        if self.ref_model is not None and getattr(self.trainer, "ref_model", None) is None:
            try:
                self.trainer.ref_model = self.ref_model
            except Exception:
                pass

        self._wire_pre_update_hooks(self.trainer)
        self._init_trackio(project_name)

        print("Starting Sebeni GRPO Training Loop (SAMPG)...")
        train_result = self.trainer.train()
        self.save_model(self.config.trainer.output_dir)

        if self.config.trainer.push_to_hub:
            self.push_to_hub(
                repo_id=self.config.trainer.hub_model_id,
                token=self.config.trainer.hub_token,
                private=self.config.trainer.hub_private_repo,
            )

        self._finish_trackio()
        return train_result
