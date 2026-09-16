import re
import json
from typing import List, Dict, Any, Optional, Callable, Tuple
from beni.core.compute.helpers import recognized_word
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute import Sentence, Token, Analysis, Morpheme
from beni.core.morphotactic import dabax
from beni.core.morphotactic.distil import distillation
from beni.core.srl.config import RewardConfig
from beni.utils import config as cfg


class RewardManager:
    """
    Reward manager implementing R_morph, R_rule, R_format, and user-customized rewards.

    All reward functions follow the TRL GRPO convention:
        f(completions, language=..., **kwargs) -> List[float]

    GRPOTrainer automatically injects extra dataset columns (e.g. "language")
    as keyword arguments, so the "language" column from format_dataset()
    flows into compute_rewards / the reward functions without any extra plumbing.
    """

    def __init__(self, reward_config: Optional[RewardConfig] = None, scorer: Optional[MorphologyScorer] = None):
        self.config = reward_config or RewardConfig()
        self.scorer = scorer or MorphologyScorer()
        self.format_scores: List[float] = []
        self.total_reward: float = 0.0
        self.predicted_stages: List[List[Any]] = []
        self.custom_rewards: Dict[str, Tuple[Callable, float]] = {}

    def register_custom_reward(self, name: str, reward_fn: Callable, weight: float = 1.0) -> None:
        """Register a custom reward function with a given weight."""
        self.custom_rewards[name] = (reward_fn, weight)

    def get_reward_functions(self) -> List[Callable]:
        """Return list of active reward functions wrapped to apply config weights."""
        funcs = []
        if self.config.enable_format_reward:
            funcs.append(self.reward_format)
        if self.config.enable_morph_reward:
            funcs.append(self.reward_morph)
        if self.config.enable_rule_reward:
            funcs.append(self.reward_rule)
        if getattr(self.config, "enable_lang_reward", False):
            funcs.append(self.reward_lang)
        for name, (fn, w) in self.custom_rewards.items():
            def make_wrapped(reward_func=fn, weight=w):
                def wrapped(*args, **kwargs):
                    scores = reward_func(*args, **kwargs)
                    return [s * weight for s in scores]
                wrapped.__name__ = reward_func.__name__
                return wrapped
            funcs.append(make_wrapped())
        return funcs

    def clear(self) -> None:
        self.format_scores.clear()
        self.total_reward = 0.0
        self.predicted_stages.clear()

    @staticmethod
    def dict_to_morpheme(d: dict) -> Morpheme:
        children = ([RewardManager.dict_to_morpheme(child) for child in d.get("morphemes", [])] if "morphemes" in d else None)

        return Morpheme(
            form=d.get("form", ""), ps=d.get("ps", []),
            gloss=d.get("gloss", ""), morphemes=children)

    @staticmethod
    def parse_json_to_sentence(json_data: dict, text: str = "") -> Sentence:
        tokens = []
        for t_data in json_data.get("tokens", []):
            analyses = []
            for a_data in t_data.get("analyses", []):
                morphemes = [RewardManager.dict_to_morpheme(m) for m in a_data.get("morphemes", [])]
                analyses.append(Analysis(
                    form=a_data.get("form", ""),
                    ps=a_data.get("ps", []),
                    gloss=a_data.get("gloss", ""),
                    morphemes=morphemes,
                ))
            tokens.append(Token(
                surface=t_data.get("surface", ""),
                stage=t_data.get("stage", -2),
                analyses=analyses,
            ))
        return Sentence(text=text, tokens=tokens)

    @staticmethod
    def extract_json(text: str) -> dict:
        text = text.strip()
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r'\{.*\}', text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {}

    def completions_to_sentences(self, completions: List[str]) -> List[Sentence]:
        """Shared parse step: raw model completions -> Sentence objects."""
        sentences = []
        for completion in completions:
            json_data = self.extract_json(completion)
            if json_data:
                sentences.append(self.parse_json_to_sentence(json_data, json_data.get("text", "")))
            else:
                sentences.append(Sentence(text="", tokens=[]))
        return sentences

    def reward_format(self, 
        completions: List[str], 
        language: Optional[List[str]] = None, **kwargs) -> List[float]:
        """R_format: Valid JSON structure with a 'tokens' list."""

        scores = []
        
        for text in completions:
            parsed = self.extract_json(text)
            valid = bool(parsed and isinstance(parsed.get("tokens"), list))
            score = self.config.format_weight if valid else 0.0
            scores.append(score)
            self.format_scores.append(score)
            self.total_reward += score

        return scores

    def reward_morph(self, 
        completions: List[str], 
        language: Optional[List[str]] = None, **kwargs) -> List[float]:
        """R_morph: Ratio of Stage 1 words over (Stage 1 + Stage >=4) words."""

        scores = []
        langs = language or [""] * len(completions)
        
        for sent, lang in zip(self.completions_to_sentences(completions), langs):
            tokens = sent.tokens
            if not tokens:
                scores.append(0.0)
                continue

            group = cfg.get_group_code(lang) if lang else lang
            distil = distillation.Distiller(lang_code=group)
            daba_x = dabax.DabaX(
                lang=group, gram=distil.gram_path, ldict=distil.dict_path)
            
            stages = []

            for tok in tokens:
                parsed_sents = daba_x.loader(tok.surface)
                parsed = parsed_sents[0] if parsed_sents else None
                if parsed and parsed.tokens:
                    stages.append(1 if parsed.tokens[0].has_valid_stage else 0)
                else:
                    stages.append(0)

            ratio = sum(stages) / len(stages)

            score = self.config.morph_weight * ratio
            scores.append(score)
            self.total_reward += score

        return scores

    def reward_rule(
        self, prompts: List[Dict[str, Any]], completions: List[str], 
        language: Optional[List[str]] = None, reference=None, **kwargs) -> List[float]:

        """R_rule: POS/valence adherence via Phi & MCS against the reference."""
        predicted = self.completions_to_sentences(completions)
        reference_sentences = self._parse_references(prompts, reference=reference)
        langs = language or [""] * len(completions)

        scores = []
        for i, (pred_sent, ref_sent, lang) in enumerate(zip(predicted, reference_sentences, langs)):
            group = cfg.get_group_code(lang) if lang else lang
            distil = distillation.Distiller(lang_code=group)
            daba_x = dabax.DabaX(lang=group, gram=distil.gram_path, ldict=distil.dict_path)

            pred_loaded = daba_x.loader(pred_sent.text) if pred_sent.text else []
            ref_loaded = daba_x.loader(ref_sent.text) if ref_sent.text else []
            pred_sent = pred_loaded[0] if pred_loaded else pred_sent
            ref_sent = ref_loaded[0] if ref_loaded else ref_sent

            phi_score = self.scorer.phi(pred_sent)
            mcs_score = self.scorer.mcs(pred_sent, ref_sent)

            score = self.config.rule_weight * 0.5 * (phi_score + mcs_score)
            scores.append(score)

            self.total_reward += score
            self.predicted_stages.append([t.stage for t in pred_sent.tokens])

        return scores

    @staticmethod
    def _parse_references(prompts: List[Dict[str, Any]], reference=None) -> List[Sentence]:
        reference_sentences = []

        if reference is not None:
            for ref in reference:
                if isinstance(ref, dict) and ref:
                    reference_sentences.append(
                        RewardManager.parse_json_to_sentence(ref, ref.get("text", "")))
                elif isinstance(ref, str) and ref:
                    parsed = RewardManager.extract_json(ref)
                    if parsed:
                        reference_sentences.append(
                            RewardManager.parse_json_to_sentence(parsed, ref))
                    else:
                        reference_sentences.append(Sentence(text="", tokens=[]))
                else:
                    reference_sentences.append(Sentence(text="", tokens=[]))
            return reference_sentences

        for prompt in prompts:
            if isinstance(prompt, dict) and ('role' in prompt):
                text = prompt.get("content", "")
                ref_json = RewardManager.extract_json(text)
                if ref_json:
                    reference_sentences.append(RewardManager.parse_json_to_sentence(ref_json, text))
                else:
                    reference_sentences.append(Sentence(text="", tokens=[]))
            else:
                reference_sentences.append(Sentence(text="", tokens=[]))
        
        return reference_sentences

    def reward_lang(
        self,
        completions: List[str],
        language: Optional[List[str]] = None,
        **kwargs,
    ) -> List[float]:
        """R_lang: language identity. Mixed or mismatched JSON ``lang`` scores 0."""
        from beni.core.language import Language

        scores = []
        langs = language or [""] * len(completions)
        weight = getattr(self.config, "lang_weight", 0.2)

        for text, lang in zip(completions, langs):
            parsed = self.extract_json(text)
            expected = Language.from_code(lang).group_code if lang else ""
            got = str(parsed.get("lang", "") or "").strip().lower()
            if not parsed or not expected or not got:
                score = 0.0
            else:
                got_group = Language.from_code(got).group_code
                score = weight if got_group == expected else 0.0
            scores.append(score)
            self.total_reward += score

        return scores

    def compute_rewards(self, completions: List[str], prompts: List[Dict[str, Any]],
                        languages: Optional[List[str]] = None) -> Dict[str, List[float]]:
        """Compute all rewards for a batch of completions."""
        format_scores = self.reward_format(completions, language=languages)
        morph_scores = self.reward_morph(completions, language=languages)
        rule_scores = self.reward_rule(completions, prompts=prompts, language=languages)
        lang_scores = self.reward_lang(completions, language=languages) if getattr(
            self.config, "enable_lang_reward", False) else [0.0] * len(completions)

        return {
            "format": format_scores,
            "morph": morph_scores,
            "rule": rule_scores,
            "lang": lang_scores,
            "total": [
                f + m + r + lg
                for f, m, r, lg in zip(format_scores, morph_scores, rule_scores, lang_scores)
            ],
        }
