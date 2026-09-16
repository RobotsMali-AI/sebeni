# `beni.core.compute`

Phi, MER, MCS, U, and the reward manager.

```python
from beni.core.compute.metrics import MorphologyScorer
from beni.core.compute.rewards import RewardManager

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
        - get_reward_functions
        - compute_rewards
      show_root_heading: true
      heading_level: 2
      show_if_no_docstring: false
