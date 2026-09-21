
"""
Morphological Analysis Scoring Metrics
======================================

Unified module for evaluating morphological decomposition quality.
Includes MER, MCS, Uncertainty, and Phi scores.

Metrics:
    - MER  : Morpheme Error Rate (edit distance at morpheme level)
    - MCS  : Morphological Composition Stage accuracy
    - U    : Token-level Uncertainty cost error (indicator + log-ratio)
    - Phi  : Corpus-level Morphological Integrity score
"""

from __future__ import annotations
import numpy as np
import warnings
from typing import Union, List, Dict
from beni.core.compute.helpers import levenshtein_matrix, _traceback_ld_ops
from beni.core import Token, Sentence


class MorphologyScorer:
    """
    Unified scorer for morphological analysis evaluation.
    """

    def __init__(self, beta: float = 0.1, eps: float = 1e-8, empr: float = 0.5):
        self.beta = beta
        self.eps = eps
        self.empr = empr

    def phi(self, sentence: Union[Sentence, List[Token]]) -> float:
        """
        Compute Phi (Morphological Integrity Score)
        Score Structure:
            - stage < 0 : 0.0 Non-Language tokens
            - 0 <= stage < 6: 1.0  (analyzed)
            - stage == 6: empr (Empirical / Emprunt/borrowing foreign token)
            - stage > 6: eps (fallback value)
            - non-numeric parser labels: 1.0
        """
        from beni.core.compute.helpers import stage_to_phi
        tokens = sentence.tokens if isinstance(sentence, Sentence) else sentence
        
        if(not tokens):
            return 0.0
        
        scores = [stage_to_phi(t.stage) for t in tokens]
        return sum(scores) / len(scores)

    def phi_corpus(self, sentences: List[Sentence]) -> Dict[str, float]:
        """ Compute phi for list of texts """
        mapping = {sent.text: self.phi(sent) for sent in sentences}
        return {
            'scores': mapping,
            'avg': float(np.mean(list(mapping.values()))) if mapping else 0.0
        }

    def mer(self, ref_tokens, hyp_tokens) -> float:
        """
        Compute Morpheme Error Rate.

        MER = (S_m + D_m + I_m) / N_m

        Parameters
        ----------
        ref_tokens : Token or list of str
            Reference morpheme sequence (or a Token whose best analysis is used).
        hyp_tokens : Token or list of str
            Hypothesis morpheme sequence.

        Returns
        -------
        float
            Edit-distance rate. 0 when both sequences are empty; inf when
            the reference is empty and the hypothesis is not.
        """
        from beni.core.compute.helpers import to_morpheme_list
        ref = to_morpheme_list(ref_tokens)
        hyp = to_morpheme_list(hyp_tokens)

        m, n = len(ref), len(hyp)

        if m == 0 and n == 0:
            return 0.0
        if m == 0:
            return float('inf')

        dp = levenshtein_matrix(ref, hyp)
        s_m, d_m, i_m = _traceback_ld_ops(dp, ref, hyp)

        return (s_m + d_m + i_m) / m

    def mer_sentence(self, ref_sentence: Sentence, hyp_sentence: Sentence, agg: str = 'mean') -> float:
        """
        agg: mean vs sum / micro
        TODO: NAIVE & Requires review
        """
        if agg == 'micro':
            refs = [
                form
                for token in ref_sentence.tokens
                for form in token.best_morphemes()
            ]
            hyps = [
                form
                for token in hyp_sentence.tokens
                for form in token.best_morphemes()
            ]
            return self.mer(refs, hyps)


        scores = [self.mer(rt, ht) for rt, ht in zip(ref_sentence.tokens, hyp_sentence.tokens)]

        if agg == 'mean':
            return np.mean(scores) if scores else 0.0
        return np.sum(scores) if scores else 0.0

    def mcs(self, predicted: Union[Sentence, List[Token]], reference: Union[Sentence, List[Token]]) -> float:
        """
        Compute Morphosyntatic Consistency Score
        MCS = (1/|T|) * Σ I(Stage_pred(w) == Stage_ref(w))
        """
        pred_tokens = predicted.tokens if isinstance(predicted, Sentence) else predicted
        ref_tokens = reference.tokens if isinstance(reference, Sentence) else reference
        
        if len(pred_tokens) != len(ref_tokens):
            # FIXME: Token capping enfored to min length(prd, ref)
            warnings.warn(f"Token count mismatch: {len(pred_tokens)} vs {len(ref_tokens)}")
        
        if not ref_tokens:
            return 0.0

        correct = sum(
            1 for pt, rt in zip(pred_tokens, ref_tokens)
            if str(pt.stage) == str(rt.stage))

        return correct / max(len(ref_tokens), len(pred_tokens))

    def mcs_corpus(self, 
                   predicted: List[Sentence], 
                   reference: List[Sentence]) -> float:
        """MCS over an entire corpus [mu-averaged]."""
        all_correct = 0
        all_total = 0

        for ps, rs in zip(predicted, reference):
            ptokens = ps.tokens if isinstance(ps, Sentence) else ps
            rtokens = rs.tokens if isinstance(rs, Sentence) else rs
            all_correct += sum(
                1 for p, r in zip(ptokens, rtokens)
                if str(p.stage) == str(r.stage)
            )
            all_total += len(ptokens)

        return all_correct / all_total if all_total > 0 else 0.0


    def uncertainty_weighted_cost_error(self, 
                    token: Token, 
                    model_prob: float, 
                    ref_prob: float) -> float:
        """
        Compute token uncertainty weighted cost error U(w_i, o_i).
         
        U = I(Stage_pred ≠ -1) + β * log((π_θ + ε) / (π_θ_ref + ε))
         
        Parameters
        ----------
        token : Token
            The token with predicted stage
        model_prob : float
            π_θ(w_i | o_i) — trained model probability
        ref_prob : float
            π_θ_ref(w_i | o_i) — reference model probability
            
        Returns
        -------
        float
            Uncertainty weighted cost error score
        """
        indicator = 1.0 if token.has_valid_stage else 0.0
        ratio = (model_prob + self.eps) / (ref_prob + self.eps)
        log_ratio = np.log(ratio)
        return indicator + self.beta * log_ratio
    
    def uncertainty_weighted_cost_error_sentence(self, 
                    sentence: Sentence, 
                    model_probs: List[float], 
                    ref_probs: List[float]) -> float:
        """
        Compute sentence uncertainty weighted cost error U(w_i, o_i).
         
        U = mean I(Stage_pred != -1) + beta * mean log((pi_theta + eps) / (pi_ref + eps))

        Morphological-analysis tokens and LM/BPE tokens are averaged separately;
        they are not positionally aligned.
         
        Parameters
        ----------
        sentence : Sentence
            The sentence with predicted stage
        model_probs : List[float]
            π_θ(w_i | o_i) — trained model probabilities
        ref_probs : List[float]
            π_θ_ref(w_i | o_i) — reference model probabilities
            
        Returns
        -------
        float
            Sentence uncertainty weighted cost error score
        """
        indicators = [1.0 if token.has_valid_stage else 0.0 for token in sentence.tokens]
        log_ratios = [(model_prob + self.eps) / (ref_prob + self.eps) for model_prob, ref_prob in zip(model_probs, ref_probs)]
        log_ratios = [np.log(lr) for lr in log_ratios]
        indicator_mean = float(np.mean(indicators)) if indicators else 0.0
        ratio_mean = float(np.mean(log_ratios)) if log_ratios else 0.0
        return indicator_mean + self.beta * ratio_mean

    def evaluate(self, 
                 predicted: Union[Sentence, List[Token]], 
                 reference: Union[Sentence, List[Token]] = None):
        """
        Compute all metrics for a single sentence or a list of sentences.
        """
        results = {
            "mer": self.mer_sentence(reference, predicted),
            "mcs": self.mcs(predicted, reference),
            "phi": self.phi(predicted)
        }

        if isinstance(predicted, list) and isinstance(reference, list):
            results['mer'] = self.mer_sentence(reference, predicted, agg='mean')
            results['mcs'] = self.mcs_corpus(predicted, reference)
            results['phi'] = self.phi_corpus(predicted)['avg']

        return results
