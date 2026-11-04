# `beni.core.compute`

Phi, MER, MCS, U (UWEC), and the reward manager. The package itself
re-exports the data classes `Morpheme`, `Analysis`, `Token`, and `Sentence`
from `beni.core`.

```python
from beni.core.compute import Sentence, Token
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager

sentence = Sentence(text="aw", lang="bam", tokens=[Token(surface="aw", stage=1)])

scorer = MorphologyScorer()
phi = scorer.phi(sentence)
mer = scorer.mer(["a", "b"], ["a", "c"])

rm = RewardManager()
fmt = rm.reward_format(['{"text": "aw", "lang": "bam", "tokens": []}'])
lang = rm.reward_lang(
    ['{"text": "aw", "lang": "bam", "tokens": []}'],
    language=["bam"],
)
```

::: beni.core.compute.metrics.MorphologyScorer
    options:
      members:
        - phi
        - phi_corpus
        - mer
        - mer_sentence
        - mcs
        - mcs_corpus
        - uncertainty_weighted_cost_error
        - uncertainty_weighted_cost_error_sentence
        - evaluate
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.rewards.RewardManager
    options:
      members:
        - reward_format
        - reward_morph
        - reward_rule
        - reward_lang
        - register_custom_reward
        - get_reward_functions
        - compute_rewards
        - completion_text
        - completions_to_sentences
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false

::: beni.core.compute.helpers
    options:
      members:
        - stage_to_phi
        - recognized_word
        - to_morpheme_list
        - levenshtein_matrix
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false
