"""Typer CLI: ``sebeni init|train|distill|eval|exp|push|generate|wordfreq``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import typer

from beni.core.language import Language, parse_lang_codes
from beni.core.srl.config import MasterConfig
from beni.utils import config as cfg

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Sebeni — self-aware morphotactic generation for extremely low-resource languages.",
)


def _load_config(
    config: Optional[Path],
    working_dir: Optional[Path] = None,
) -> MasterConfig:
    if config is not None:
        return MasterConfig.from_yaml(config, working_dir=working_dir)
    mc = MasterConfig()
    mc.apply_workdir(cli_path=working_dir)
    return mc


def _records(config: MasterConfig):
    from beni.data.datasets import SebeniDataLoader

    source = config.data.source
    loader = SebeniDataLoader(config.data)
    if source:
        return loader.load(source)
    return loader.load_sebeni()


def _records_by_language(mc: MasterConfig, records) -> Dict[str, List[str]]:
    from beni.core.srl.algorithm1 import group_texts_by_language, records_text_langs

    default_lang = mc.data.default_lang or "bam"
    grouped = group_texts_by_language(
        records_text_langs(records, default_lang),
        default_lang=default_lang,
    )
    if mc.data.languages:
        allowed = set(mc.languages())
        grouped = {k: v for k, v in grouped.items() if k in allowed}
    return grouped


def _distiller_for(mc: MasterConfig, group: str):
    from beni.core.morphotactic.distil.distillation import Distiller

    distiller = Distiller(
        lang_code=group,
        backend=mc.distillation.selected_backend,
        model=mc.distillation.model,
        working_dir=mc.distillation.working_dir or mc.working_dir,
        vertex=mc.distillation.vertex,
        base_url=mc.distillation.base_url,
        gguf_path=mc.distillation.gguf_path,
        n_ctx=mc.distillation.n_ctx,
        max_input_chars=mc.distillation.max_input_chars,
    )
    distiller.handle_baselines()
    return distiller


def _group_codes_from_records(records) -> List[str]:
    seen: List[str] = []
    for rec in records or []:
        if isinstance(rec, dict):
            lang = rec.get("lang") or rec.get("language") or "bam"
        else:
            lang = getattr(rec, "lang", None) or "bam"
        group = Language.from_code(str(lang)).group_code
        if group not in seen:
            seen.append(group)
    return seen


def _packaged_exp_yaml() -> Path:
    pkg = Path(__file__).resolve().parents[1] / "data" / "exp.yaml"
    repo = Path(__file__).resolve().parents[2] / "configs" / "exp.yaml"
    if repo.is_file():
        return repo
    return pkg


def emit_kveritas_metrics(report: Dict, step: int = 0) -> None:
    """Print stdout lines K-Veritas already understands."""
    phi = report.get("phi")
    if phi is not None:
        typer.echo(f"KVERITAS_METRIC name=phi value={float(phi):.6g} step={step}")
    for lang, row in (report.get("by_language") or {}).items():
        val = (row or {}).get("phi")
        if val is None:
            continue
        typer.echo(f"KVERITAS_METRIC name=phi_{lang} value={float(val):.6g} step={step}")


def maybe_kveritas_seal(output: Path) -> None:
    import shutil
    import subprocess

    binary = shutil.which("kveritas")
    if not binary:
        typer.echo(
            "kveritas not on PATH; skip seal. See https://kveritas.org/docs",
            err=True,
        )
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    result = subprocess.run([binary, "seal", "--output", str(output)], check=False)
    if result.returncode != 0:
        typer.echo(f"kveritas seal exited {result.returncode}", err=True)


def _evaluate(mc: MasterConfig, records) -> Tuple[Dict, Path]:
    """Score Φ per language; write ``{working_dir}/exp/eval.json`` and a safety snapshot."""
    from beni.core.compute.metrics import MorphologyScorer
    from beni.core.morphotactic.dabax import get_dabax
    from beni.core.safety.governor import SafetyGovernor

    grouped = _records_by_language(mc, records)
    scorer = MorphologyScorer()
    by_language = {}
    total_sent = 0
    weighted_phi = 0.0
    for group, texts in grouped.items():
        distiller = _distiller_for(mc, group)
        dabax = get_dabax(
            group,
            gram=distiller.gram_path,
            ldict=distiller.dict_path,
            process=True,
            runtime_dir=cfg.get_workdir().runtime,
        )
        sentences = []
        for text in texts:
            try:
                sentences.extend(dabax.loader(text) or [])
            except Exception:
                continue
        phi = scorer.phi_corpus(sentences)["avg"] if sentences else 0.0
        n = len(sentences)
        by_language[group] = {
            "phi": phi,
            "n_sentences": n,
            "checkpoint_id": distiller.checkpoint_id(),
            "language": group,
        }
        total_sent += n
        weighted_phi += phi * n
    avg_phi = (weighted_phi / total_sent) if total_sent else 0.0
    langs = list(grouped.keys()) or mc.languages()
    report = {
        "phi": avg_phi,
        "tau": mc.distillation.tau,
        "n_sentences": total_sent,
        "languages": langs,
        "by_language": by_language,
        "algorithm": mc.algorithm,
    }
    out = cfg.get_workdir().exp / "eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    gov = SafetyGovernor(mc.safety.to_spec(tau=mc.distillation.tau, kl_beta=mc.trainer.beta))
    gov.record_snapshot(
        phi=avg_phi,
        tau=mc.distillation.tau,
        checkpoint_id=",".join(v["checkpoint_id"] or "" for v in by_language.values()),
        algorithm=mc.algorithm,
        language=",".join(langs),
        group_code=",".join(langs),
        extra={"languages": langs, "by_language": by_language},
    )
    gov.write_snapshot(cfg.get_workdir().models)
    return report, out


def default_config_yaml(langs: List[str], working_dir: str) -> str:
    """Scaffold YAML for ``sebeni init``."""
    codes = Language.group_codes(langs) or ["bam"]
    default = codes[0]
    langs_yaml = "[" + ", ".join(codes) + "]"
    slug = "-".join(codes)
    return f"""# Sebeni MasterConfig — https://seben.robotsmali.org/docs
# working_dir is relocatable: CLI -w and SEBENI_HOME / SEBENI_WORKING_DIR also apply.
# Alignment is multilingual: each row keeps its language; Φ / Distiller / {{G, D}}
# are per group code (MKU not MLQ). Tune trainer/model hyperparams below or via
# `sebeni train --lr ...` (see docs/hyperparams.md).
# algorithm: grpo | dpo | apo  (policy-update plugins; SAMPG Φ / Distiller is unchanged)
project_name: sebeni-{slug}
algorithm: grpo          # grpo | dpo | apo
working_dir: {working_dir}

model:
  model_name: HuggingFaceTB/SmolLM2-135M
  # ref_model_name: null
  load_in_4bit: true
  use_peft: true
  lora_r: 16
  lora_alpha: 32
  lora_dropout: 0.1
  # lora_target_modules: [q_proj, v_proj, k_proj, o_proj]

data:
  default_lang: {default}
  languages: {langs_yaml}
  source: null             # path, glob, list of paths, or Hugging Face dataset id
  scheme: completion       # completion | preference | online_group

trainer:
  framework: torch         # torch | jax
  learning_rate: 5.0e-6
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 10
  num_train_epochs: 1.0
  logging_steps: 1
  save_steps: 50
  max_grad_norm: 0.1
  beta: 0.1
  num_generations: 4
  num_iterations: 1
  temperature: 0.9
  top_p: 1.0
  top_k: 50
  warmup_ratio: 0.0
  warmup_steps: 0
  weight_decay: 0.0
  lr_scheduler_type: cosine
  seed: 42
  optim: adamw_torch
  bf16: false
  fp16: false
  gradient_checkpointing: false
  dataloader_num_workers: 0
  use_cpu: false
  # output_dir defaults to {{working_dir}}/models

distillation:
  enabled: true
  backend: algorithmic     # algorithmic | gguf | google | openai | groq | together
  model: gemini-2.5-flash
  tau: 0.5
  hitl: false

wordfreq:
  raw_inputs: null         # defaults to packaged beni/data/raw

reward:
  format_weight: 0.2
  morph_weight: 0.4
  rule_weight: 0.4
  lang_weight: 0.2

safety:
  enabled: true
  require_model_card: true
  require_safety_snapshot: true
"""


@app.command()
def init(
    lang: List[str] = typer.Option(
        ["bam"],
        "--lang",
        help="ISO or Sebeni group code. Repeat or comma-separate for multilingual (e.g. --lang bam --lang mku).",
    ),
    working_dir: Path = typer.Option(
        Path("./runs/sebeni-001"),
        "-w",
        "--working-dir",
        help="Run directory (created if missing).",
    ),
):
    """Write config.yaml and the workdir layout (data/, models/, runs/, exp/, runtime/)."""
    codes = Language.group_codes(parse_lang_codes(lang)) or ["bam"]
    wd = cfg.set_working_dir(working_dir, ensure=True)
    config_path = Path(wd.root) / "config.yaml"
    rel = str(wd.root)
    config_path.write_text(default_config_yaml(codes, rel), encoding="utf-8")
    typer.echo(f"Wrote {config_path}")
    typer.echo(f"Working dir {wd.root}")
    typer.echo("Languages: " + ", ".join(codes))
    typer.echo("Next: sebeni train -c config.yaml")


@app.command()
def train(
    config: Path = typer.Option(..., "-c", "--config", help="YAML or JSON MasterConfig."),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(
        None,
        "--lang",
        help="Override data.languages (repeatable / comma-separated).",
    ),
    hitl: Optional[bool] = typer.Option(None, "--hitl/--no-hitl", help="HITL on initial distill."),
    lr: Optional[float] = typer.Option(None, "--lr", help="Optimizer learning rate."),
    batch_size: Optional[int] = typer.Option(
        None, "--batch-size", "--per-device-train-batch-size", help="Per-device train batch size."
    ),
    grad_accum: Optional[int] = typer.Option(
        None, "--grad-accum", "--gradient-accumulation-steps"
    ),
    max_steps: Optional[int] = typer.Option(None, "--max-steps"),
    epochs: Optional[float] = typer.Option(None, "--epochs", "--num-train-epochs"),
    beta: Optional[float] = typer.Option(None, "--beta", help="KL / DPO β."),
    num_generations: Optional[int] = typer.Option(None, "--num-generations", help="GRPO group size G."),
    max_prompt_length: Optional[int] = typer.Option(None, "--max-prompt-length"),
    max_completion_length: Optional[int] = typer.Option(None, "--max-completion-length"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    warmup_ratio: Optional[float] = typer.Option(None, "--warmup-ratio"),
    warmup_steps: Optional[int] = typer.Option(None, "--warmup-steps"),
    weight_decay: Optional[float] = typer.Option(None, "--weight-decay"),
    seed: Optional[int] = typer.Option(None, "--seed"),
    save_steps: Optional[int] = typer.Option(None, "--save-steps"),
    optim: Optional[str] = typer.Option(None, "--optim"),
    lr_scheduler: Optional[str] = typer.Option(None, "--lr-scheduler", help="e.g. cosine, linear."),
    lora_r: Optional[int] = typer.Option(None, "--lora-r"),
    lora_alpha: Optional[int] = typer.Option(None, "--lora-alpha"),
    lora_dropout: Optional[float] = typer.Option(None, "--lora-dropout"),
    load_in_4bit: Optional[bool] = typer.Option(None, "--load-in-4bit/--no-load-in-4bit"),
    use_peft: Optional[bool] = typer.Option(None, "--peft/--no-peft"),
    bf16: Optional[bool] = typer.Option(None, "--bf16/--no-bf16"),
    fp16: Optional[bool] = typer.Option(None, "--fp16/--no-fp16"),
    gradient_checkpointing: Optional[bool] = typer.Option(
        None, "--grad-checkpoint/--no-grad-checkpoint"
    ),
    use_cpu: Optional[bool] = typer.Option(None, "--use-cpu/--no-use-cpu"),
):
    """Run SAMPG (Φ / τ / Distiller) then the configured policy-update plugin."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(
        languages=lang,
        learning_rate=lr,
        batch_size=batch_size,
        grad_accum=grad_accum,
        max_steps=max_steps,
        epochs=epochs,
        beta=beta,
        num_generations=num_generations,
        max_prompt_length=max_prompt_length,
        max_completion_length=max_completion_length,
        temperature=temperature,
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        seed=seed,
        save_steps=save_steps,
        optim=optim,
        use_cpu=use_cpu,
        lora_r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        load_in_4bit=load_in_4bit,
        use_peft=use_peft,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=gradient_checkpointing,
        lr_scheduler_type=lr_scheduler,
        hitl=hitl,
    )
    from beni.core.srl.unified import SRLTrainer

    records = _records(mc)
    trainer = SRLTrainer(mc)
    trainer.train(records)


@app.command()
def distill(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
    hitl: Optional[bool] = typer.Option(None, "--hitl/--no-hitl"),
):
    """Baseline-only Distiller job (train does this automatically when needed)."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang, hitl=hitl)
    from beni.core.srl.grpo.grpo import SebeniGrpo

    records = _records(mc)
    SebeniGrpo(mc).run_batch_distillation(records)


@app.command()
def eval(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
):
    """Score Φ / MER / MCS **per language**; write ``{working_dir}/exp/eval.json``."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang)
    _report, out = _evaluate(mc, _records(mc))
    typer.echo(str(out))


def _run_exp(config: Optional[Path], working_dir: Optional[Path]) -> None:
    cfg_path = config or _packaged_exp_yaml()
    if not Path(cfg_path).is_file():
        typer.echo(f"experiment config not found: {cfg_path}", err=True)
        raise typer.Exit(code=2)
    mc = _load_config(Path(cfg_path), working_dir)
    mc.data.source = None
    mc.data.languages = None
    from beni.data.datasets import SebeniDataLoader
    from beni.core.srl.unified import SRLTrainer

    loader = SebeniDataLoader(mc.data)
    train_records = loader.load_sebeni(split="train")
    if not train_records:
        typer.echo("Packaged train split is empty (beni/data/raw).", err=True)
        raise typer.Exit(code=1)
    langs = _group_codes_from_records(train_records)
    mc.data.languages = langs
    mc.data.known_langs = set(langs)
    if langs:
        mc.data.default_lang = langs[0]
    trainer = SRLTrainer(mc)
    trainer.train(train_records)

    mc.data.languages = None
    test_records = SebeniDataLoader(mc.data).load_sebeni(split="test")
    report, out = _evaluate(mc, test_records)
    typer.echo(str(out))
    if mc.experiment.kveritas:
        emit_kveritas_metrics(report)
    if mc.experiment.kveritas_seal:
        maybe_kveritas_seal(cfg.get_workdir().exp / "report.pdf")


@app.command("exp")
def exp(
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="YAML MasterConfig. Defaults to packaged exp.yaml; data.source is ignored.",
    ),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
):
    """Train one multilingual SAMPG policy on packaged raw data, then eval test.json."""
    _run_exp(config, working_dir)


@app.command("experiment")
def experiment(
    config: Optional[Path] = typer.Option(
        None,
        "-c",
        "--config",
        help="YAML MasterConfig. Defaults to packaged exp.yaml; data.source is ignored.",
    ),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
):
    """Alias of ``sebeni exp``."""
    _run_exp(config, working_dir)


@app.command()
def push(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    repo_id: Optional[str] = typer.Option(None, "--repo-id"),
):
    """Push ``output_dir`` to the Hub (requires model card + safety snapshot)."""
    mc = _load_config(config, working_dir)
    from beni.core.srl.unified import SRLTrainer

    trainer = SRLTrainer(mc)
    trainer.load_models()
    rid = repo_id or mc.trainer.hub_model_id
    trainer.push_to_hub(repo_id=rid, token=mc.trainer.hub_token, private=mc.trainer.hub_private_repo)


@app.command()
def generate(
    config: Path = typer.Option(..., "-c", "--config"),
    prompt: str = typer.Option(..., "--prompt"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    max_length: int = typer.Option(128, "--max-length"),
    lang: Optional[str] = typer.Option(None, "--lang", help="Expected JSON lang for R_lang (one row)."),
):
    """Generate from the saved policy, gated by SafetyGovernor (format / R_lang)."""
    mc = _load_config(config, working_dir)
    from beni.core.srl.unified import SRLTrainer
    from beni.core.compute.rewards import RewardManager

    trainer = SRLTrainer(mc)
    trainer.load_models()
    text = trainer.generate(prompt, max_length=max_length)
    rm = RewardManager(reward_config=mc.reward)
    fmt = rm.reward_format([text])
    expected = lang or (mc.languages()[0] if mc.languages() else "bam")
    lang_scores = rm.reward_lang([text], language=[expected])
    if fmt[0] == 0.0 and mc.safety.format_invalid_blocks_update:
        typer.echo("Warning: completion failed R_format (JSON + tokens).", err=True)
    if lang_scores[0] == 0.0 and mc.safety.require_language:
        typer.echo(f"Warning: completion failed R_lang (expected {expected}).", err=True)
    typer.echo(text)


@app.command()
def wordfreq(
    config: Path = typer.Option(..., "-c", "--config"),
    working_dir: Optional[Path] = typer.Option(None, "-w", "--working-dir"),
    lang: Optional[List[str]] = typer.Option(None, "--lang"),
):
    """Build DabaX frequency maps from configured raw text inputs."""
    mc = _load_config(config, working_dir)
    mc.apply_cli_overrides(languages=lang)
    from beni.core.wordfreq import count_raw_inputs

    raw_inputs = mc.wordfreq.raw_inputs or mc.data.source or (cfg.DATA_DIR / "raw")
    base = Path(getattr(mc, "_config_file_dir", Path.cwd()))
    values = raw_inputs if isinstance(raw_inputs, list) else [raw_inputs]
    raw_inputs = [
        str(Path(value) if Path(value).is_absolute() else base / str(value))
        for value in values
    ]
    reports = count_raw_inputs(
        raw_inputs,
        languages=mc.data.languages,
        default_lang=mc.data.default_lang or "bam",
        encoding=mc.data.encoding,
    )
    root = cfg.get_workdir().exp / "wordfreq"
    index = {"languages": list(reports.keys()), "by_language": {}}
    last_path = root
    for group, report in reports.items():
        path = report.write(root / group)
        last_path = path
        index["by_language"][group] = {
            "path": str(path),
            "n_tokens": report.n_tokens,
            "n_sentences": report.n_sentences,
            "checkpoint_id": report.checkpoint_id,
            "n_misses": sum(report.misses.values()),
        }
    root.mkdir(parents=True, exist_ok=True)
    summary = root / "wordfreq.json"
    summary.write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    typer.echo(str(summary if reports else last_path))
