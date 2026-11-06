"""Reaction experiment; use --help for local smoke tests and cluster runs."""
import argparse
import json
import os
import re
from pathlib import Path
import torch
import pandas as pd
from tqdm.auto import tqdm
from transformers import AutoConfig, AutoTokenizer, AutoModelForCausalLM

# DEFAULT_MODEL = "meta-llama/Llama-4-Scout-17B-16E"
DEFAULT_MODEL = "google/gemma-3-12b-pt"
SMOKE_MODEL = "sshleifer/tiny-gpt2"

REACTIONS = [
    "Do nothing",
    "Stay away from the actor",
    "Gossip or complain to others",
    "Verbally confront the actor",
    "Physically confront the actor",
    "Inform the authorities",
    "Support the actor",
    "Praise the actor",
]

# Prompt
PROMPT_TEMPLATE = """Possible reactions: Do nothing | Stay away from the actor | Gossip or complain to others | Verbally confront the actor | Physically confront the actor | Inform the authorities | Support the actor | Praise the actor
Choose exactly one of the reactions above.

Example:

Behavior:
<behavior description>

<relationship> {mode}:
<one reaction from the list above>

###

Behavior:
{stimulus}

{relationship} {mode}:"""

QUESTION_CONFIGS = [
    ("A friend", "would", "friends_reaction"),
    ("A friend", "should", "friends_reaction_should"),
    ("An acquaintance", "would", "acquaintances_reaction"),
    ("An acquaintance", "should", "acquaintances_reaction_should"),
    ("A stranger", "would", "strangers_reaction"),
    ("A stranger", "should", "strangers_reaction_should"),
]

# Function to extract exactly one valid reaction from the generated continuation
def extract_reaction(generated_text):
    """Extract the first complete reaction if generation begins with one."""
    if generated_text is None:
        return None

    text = str(generated_text).strip()

    text = re.sub(r"^(?:answer|reaction)\s*:\s*", "", text, flags=re.I)
    text = re.sub(r"^\d+[.)]\s*", "", text)

    for reaction in REACTIONS:
        if re.match(
            rf"^{re.escape(reaction)}(?:\s|$|[.!])",
            text,
            flags=re.I
        ):
            return reaction

    return None

# Function to generate one reaction for one stimulus/question combination
def generate_reaction(
    stimulus,
    relationship,
    mode,
    processor,
    model,
    max_new_tokens=32,
    prompt_format="plain",
    fail_fast=True,
):
    if pd.isna(stimulus) or not str(stimulus).strip():
        return {
            "reaction": None,
            "raw_output": None,
            "error": "Missing stimulus",
        }

    stimulus = str(stimulus).strip()

    prompt = PROMPT_TEMPLATE.format(
        stimulus=stimulus,
        relationship=relationship,
        mode=mode,
    )

    try:
        if prompt_format == "chat":
            inputs = processor.apply_chat_template(
                [{"role": "user", "content": prompt}],
                add_generation_prompt=True,
                tokenize=True,
                return_dict=True,
                return_tensors="pt",
            )
        else:
            inputs = processor(prompt, return_tensors="pt")

        # Match the embedding device when Accelerate shards a model across GPUs.
        device = model.get_input_embeddings().weight.device
        if device.type == "meta":
            device = model.device
        inputs = inputs.to(device)

        input_length = inputs["input_ids"].shape[1]

        # Generate
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=(processor.pad_token_id if processor.pad_token_id is not None
                              else processor.eos_token_id),
            )

        # Decode newly generated tokens
        generated_tokens = outputs[0, input_length:]

        raw_output = processor.decode(
            generated_tokens,
            skip_special_tokens=True,
        ).strip()

        reaction = extract_reaction(raw_output)

        return {
            "reaction": reaction,
            "raw_output": raw_output,
            "error": None if reaction else "No single complete valid reaction",
        }

    except Exception as exc:
        if fail_fast:
            raise
        return {
            "reaction": None,
            "raw_output": None,
            "error": str(exc),
        }

# Function to convert one row with male/female stimulus variants into two rows
def reshape_gender_variants(
    df,
    id_col="ID",
    male_stimulus_col="stimuli-male",
    male_actor_col="norm-target in stimuli-male",
    female_stimulus_col="stimuli-female",
    female_actor_col="norm-target in stimuli-female",
):
    required_columns = [
        male_stimulus_col,
        male_actor_col,
        female_stimulus_col,
        female_actor_col,
    ]

    missing = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise KeyError(
            "The following required columns are missing: "
            + ", ".join(missing)
        )

    rows = []

    for fallback_id, (_, row) in enumerate(df.iterrows()):

        row_id = (
            row[id_col]
            if id_col in df.columns
            else fallback_id
        )

        # Male stimulus version
        rows.append(
            {
                "ID": row_id,
                "stimuli": row[male_stimulus_col],
                "actor": row[male_actor_col],
                "gender": "male",
            }
        )

        # Female stimulus version
        rows.append(
            {
                "ID": row_id,
                "stimuli": row[female_stimulus_col],
                "actor": row[female_actor_col],
                "gender": "female",
            }
        )

    return pd.DataFrame(rows, columns=["ID", "stimuli", "actor", "gender"])

def run_base_full(
    df,
    processor,
    model,
    id_col="ID",
    male_stimulus_col="stimuli-male",
    male_actor_col="norm-target in stimuli-male",
    female_stimulus_col="stimuli-female",
    female_actor_col="norm-target in stimuli-female",
    max_new_tokens=32,
    save_every=50,
    output_dir='.',
    out_csv="gemma_base_responses.csv",
    out_parquet="gemma_base_responses.parquet",
    include_raw_outputs=True,
    prompt_format="plain",
    fail_fast=True,
):
    """
    Run a base language model on male and female stimulus variants.

    For each original row, this function creates:
        1. one male-variant output row
        2. one female-variant output row

    Each output row contains six reaction judgments:
        - friends_reaction
        - friends_reaction_should
        - acquaintances_reaction
        - acquaintances_reaction_should
        - strangers_reaction
        - strangers_reaction_should

    Parameters
    ----------
    df : pd.DataFrame
        Wide-format input containing male and female stimulus versions.

    processor
        Hugging Face processor.

    model
        Hugging Face model.

    save_every : int or None
        Save checkpoint after this many processed stimulus variants.
        Set to None or 0 to disable checkpoint saving.

    output_dir : str
        Directory where checkpoints and final files are saved.

    include_raw_outputs : bool
        Whether to preserve raw model generations.

    Returns
    -------
    pd.DataFrame
        Long-format output with one row per gender variant.
    """

    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    model.eval()

    
    # Reshape male/female variants
    long_df = reshape_gender_variants(
        df=df,
        id_col=id_col,
        male_stimulus_col=male_stimulus_col,
        male_actor_col=male_actor_col,
        female_stimulus_col=female_stimulus_col,
        female_actor_col=female_actor_col,
    )

    print(
        f"Original rows: {len(df):,}\n"
        f"Stimulus variants: {len(long_df):,}\n"
        f"Total generations expected: "
        f"{len(long_df) * len(QUESTION_CONFIGS):,}"
    )

    results = []

    # Process every male/female stimulus variant
    for position, row in tqdm(
        long_df.iterrows(),
        total=len(long_df),
        desc="Processing stimulus variants",
    ):

        result = {
            "ID": row["ID"],
            "stimuli": row["stimuli"],
            "actor": row["actor"],
            "gender": row["gender"],
        }

        errors = []

        # Ask all six questions
        for relationship, mode, output_column in QUESTION_CONFIGS:

            response = generate_reaction(
                stimulus=row["stimuli"],
                relationship=relationship,
                mode=mode,
                processor=processor,
                model=model,
                max_new_tokens=max_new_tokens,
                prompt_format=prompt_format,
                fail_fast=fail_fast,
            )

            result[output_column] = response["reaction"]

            if include_raw_outputs:
                result[f"{output_column}_raw"] = (
                    response["raw_output"]
                )

            if response["error"]:
                errors.append(
                    f"{relationship}_{mode}: "
                    f"{response['error']}"
                )

        result["error"] = (
            " | ".join(errors)
            if errors
            else None
        )

        results.append(result)

        # Periodic checkpoint
        if save_every and len(results) % save_every == 0:

            checkpoint_df = pd.DataFrame(results)

            if out_csv:
                checkpoint_csv = os.path.join(
                    output_dir,
                    out_csv,
                )

                checkpoint_df.to_csv(
                    checkpoint_csv,
                    index=False,
                )

            if out_parquet:
                checkpoint_df.to_parquet(
                    os.path.join(output_dir, out_parquet), index=False,
                )
            print(
                f"\nCheckpoint saved after {len(results):,}/"
                f"{len(long_df):,} stimulus variants.", flush=True,
            )

    # Final dataframe
    results_df = pd.DataFrame(results)

    # Put metadata columns first
    front_columns = [
        column
        for column in [
            "ID",
            "stimuli",
            "actor",
            "gender",
        ]
        if column in results_df.columns
    ]

    remaining_columns = [
        column
        for column in results_df.columns
        if column not in front_columns
    ]

    results_df = results_df[
        front_columns + remaining_columns
    ]

    # Save
    if out_csv:

        csv_path = os.path.join(
            output_dir,
            out_csv,
        )

        results_df.to_csv(
            csv_path,
            index=False,
        )

        print(
            f"Saved CSV to: {csv_path}",
            flush=True,
        )

    if out_parquet:

        parquet_path = os.path.join(
            output_dir,
            out_parquet,
        )

        results_df.to_parquet(
            parquet_path,
            index=False,
        )

        print(
            f"Saved Parquet to: {parquet_path}",
            flush=True,
        )

    return results_df

def load_model(model_id, device, dtype, cache_dir=None):
    from transformers import AutoProcessor, AutoModelForMultimodalLM

    processor = AutoProcessor.from_pretrained(
        model_id,
        cache_dir=cache_dir,
    )

    model = AutoModelForMultimodalLM.from_pretrained(
        model_id,
        cache_dir=cache_dir,
        dtype=torch.bfloat16,
        device_map=device,
    )

    model.eval()

    return processor.tokenizer, model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--input", type=Path, default=Path(__file__).with_name("rotexamples_07072025.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("results"))
    parser.add_argument("--limit", type=int, help="Number of original rows (12 generations per row)")
    parser.add_argument("--smoke-test", action="store_true", help="Use a tiny public pipeline-test model and one input row")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--dtype", choices=["auto", "float32", "float16", "bfloat16"], default="auto")
    parser.add_argument("--prompt-format", choices=["auto", "plain", "chat"], default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--save-every", type=int, default=50)
    parser.add_argument("--no-parquet", action="store_true")
    parser.add_argument("--continue-on-error", action="store_true", help="Record inference exceptions instead of stopping")
    parser.add_argument("--cache-dir", type=Path)
    args = parser.parse_args()
    if args.smoke_test:
        args.model = SMOKE_MODEL
        args.limit = args.limit if args.limit is not None else 1
        if args.output_dir == Path("results"):
            args.output_dir = Path("results/smoke")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    if args.max_new_tokens < 1 or args.save_every < 0:
        parser.error("--max-new-tokens must be positive and --save-every nonnegative")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA requested but unavailable in this PyTorch installation")
    df = pd.read_csv(args.input, nrows=args.limit)
    reshape_gender_variants(df)  # Validate columns before downloading model weights.
    if df.empty:
        parser.error("Input contains no data rows")
    if not args.no_parquet:
        import pyarrow  # Fail before inference if the Parquet dependency is missing.
    tokenizer, model = load_model(args.model, args.device, args.dtype, args.cache_dir)
    prompt_format = args.prompt_format
    if prompt_format == "auto":
        prompt_format = "chat" if tokenizer.chat_template else "plain"
        # Preserve the original experiment's Base checkpoint and completion mode.
        if args.model == DEFAULT_MODEL:
            prompt_format = "plain"
    if prompt_format == "chat" and not tokenizer.chat_template:
        parser.error("This tokenizer has no chat template; use --prompt-format plain")
    print(f"Model: {args.model}; prompt format: {prompt_format}; device map: {getattr(model, "hf_device_map", {"model": str(model.device)})}", flush=True)
    results = run_base_full(
        df, tokenizer, model,
        output_dir=args.output_dir,
        out_parquet=None if args.no_parquet else "gemma_base_responses.parquet",
        max_new_tokens=args.max_new_tokens,
        save_every=args.save_every,
        prompt_format=prompt_format,
        fail_fast=not args.continue_on_error,
    )
    columns = [c for _, _, c in QUESTION_CONFIGS]
    valid = int(results[columns].notna().sum().sum())
    total = len(results) * len(columns)
    import transformers
    summary = {
        "model": args.model, "prompt_format": prompt_format,
        "torch": torch.__version__, "transformers": transformers.__version__,
        "device_map": {k: str(v) for k, v in getattr(model, "hf_device_map", {"model": str(model.device)}).items()},
        "dtype": str(model.dtype), "input_rows": len(df),
        "variant_rows": len(results), "generations": total,
        "valid_reactions": valid, "invalid_reactions": total - valid,
        "max_new_tokens": args.max_new_tokens,
    }
    (args.output_dir / "run_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"DONE: {valid}/{total} valid reactions. Inspect raw outputs and error columns.")
    return 0 if valid == total else 2


if __name__ == "__main__":
    raise SystemExit(main())
