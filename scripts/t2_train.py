#!/usr/bin/env python3
"""T2-00 — treino do candidato nano (<=100M) no LeanDojo Benchmark 4.

Treina um seq2seq T5 do zero (byte-level ByT5) para prever a primeira tática a
partir do estado de prova. Orçamento T2-00 congelado em
research/lean/ltp/T2-00-protocol.json.
"""

from __future__ import annotations

import argparse
import gc
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import AutoTokenizer, T5Config, T5ForConditionalGeneration

ROOT = Path(__file__).resolve().parents[1]
TRAIN_FILE = ROOT / "data" / "raw" / "leandojo4" / "extracted" / "leandojo_benchmark_4" / "random" / "train.json"
TOKENIZER_DIR = ROOT / "data" / "raw" / "reprover" / "leandojo-lean4-tacgen-byt5-small" / "67a2c53cc36186fe8539d0a342fc42c50edc68fd"
IGNORE_INDEX = -100


def load_pairs(limit: int, all_steps: bool = False) -> list[tuple[str, str]]:
    data = json.loads(TRAIN_FILE.read_text(encoding="utf-8"))
    pairs: list[tuple[str, str]] = []
    for entry in data:
        tactics = entry.get("traced_tactics") or []
        if not tactics:
            continue
        steps = tactics if all_steps else tactics[:1]
        for tactic in steps:
            state = tactic.get("state_before", "")
            text = tactic.get("tactic", "")
            if state and text:
                pairs.append((state, text))
                if len(pairs) >= limit:
                    break
        if len(pairs) >= limit:
            break
    del data
    gc.collect()
    return pairs


class TacticDataset(torch.utils.data.Dataset):
    def __init__(self, pairs: list[tuple[str, str]], tokenizer, max_input: int, max_target: int):
        self.inputs = [item[0] for item in pairs]
        self.targets = [item[1] for item in pairs]
        self.tokenizer = tokenizer
        self.max_input = max_input
        self.max_target = max_target

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, index: int) -> dict:
        encoded = self.tokenizer(
            self.inputs[index], max_length=self.max_input, truncation=True, padding=False
        )["input_ids"]
        labels = self.tokenizer(
            self.targets[index], max_length=self.max_target, truncation=True, padding=False
        )["input_ids"]
        return {"input_ids": encoded, "labels": labels}


def collate(batch: list[dict], pad_id: int) -> dict:
    max_in = max(len(item["input_ids"]) for item in batch)
    max_out = max(len(item["labels"]) for item in batch)
    input_ids, attention, labels = [], [], []
    for item in batch:
        pad_in = max_in - len(item["input_ids"])
        pad_out = max_out - len(item["labels"])
        input_ids.append(item["input_ids"] + [pad_id] * pad_in)
        attention.append([1] * len(item["input_ids"]) + [0] * pad_in)
        labels.append(item["labels"] + [IGNORE_INDEX] * pad_out)
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention),
        "labels": torch.tensor(labels),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="T2-00 treino nano")
    parser.add_argument("--model-dir", required=True)
    parser.add_argument("--examples", type=int, default=20000)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--max-input", type=int, default=512)
    parser.add_argument("--max-target", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--max-steps", type=int, default=0)
    parser.add_argument("--all-steps", action="store_true")
    parser.add_argument("--accum", type=int, default=1)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1024)
    parser.add_argument("--layers", type=int, default=4)
    parser.add_argument("--decoder-layers", type=int, default=4)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained(str(TOKENIZER_DIR))
    vocab_size = len(tokenizer)
    config = T5Config(
        vocab_size=vocab_size,
        d_model=args.d_model,
        d_ff=args.d_ff,
        d_kv=args.d_model // args.heads,
        num_layers=args.layers,
        num_decoder_layers=args.decoder_layers,
        num_heads=args.heads,
        decoder_start_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.pad_token_id,
    )
    model = T5ForConditionalGeneration(config).to(device)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    weight_bytes = sum(parameter.numel() * parameter.element_size() for parameter in model.parameters())
    pairs = load_pairs(args.examples, all_steps=args.all_steps)
    dataset = TacticDataset(pairs, tokenizer, args.max_input, args.max_target)
    loader = DataLoader(
        dataset,
        batch_size=args.batch,
        shuffle=True,
        collate_fn=lambda batch: collate(batch, tokenizer.pad_token_id),
        generator=torch.Generator().manual_seed(args.seed),
    )
    optimizer = AdamW(model.parameters(), lr=args.lr)
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()
    started = time.monotonic()
    losses = []
    step = 0
    micro = 0
    model.train()
    total_steps = math.ceil(len(loader) / args.accum) * args.epochs
    optimizer.zero_grad()
    for _ in range(args.epochs):
        for batch in loader:
            batch = {key: value.to(device) for key, value in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss / args.accum
            loss.backward()
            losses.append(outputs.loss.item())
            micro += 1
            if micro % args.accum == 0:
                optimizer.step()
                optimizer.zero_grad()
                step += 1
                if step % 50 == 0 or step == total_steps:
                    print(f"passo {step}/{total_steps} loss {sum(losses[-50:]) / len(losses[-50:]):.4f}", flush=True)
                if args.max_steps and step >= args.max_steps:
                    break
        if args.max_steps and step >= args.max_steps:
            break
    if micro % args.accum:
        optimizer.step()
        optimizer.zero_grad()
        step += 1
    wall = time.monotonic() - started
    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(model_dir)
    peak_vram = torch.cuda.max_memory_allocated() if device == "cuda" else 0
    payload = {
        "phase": "T2-00",
        "device": device,
        "examples": len(pairs),
        "epochs": args.epochs,
        "steps": step,
        "final_loss_mean_50": round(sum(losses[-50:]) / len(losses[-50:]), 4) if losses else None,
        "loss_first_50": round(sum(losses[:50]) / len(losses[:50]), 4) if losses else None,
        "parameters": parameters,
        "weights_fp32_bytes": weight_bytes,
        "wall_seconds": round(wall, 3),
        "peak_vram_bytes": peak_vram,
        "model_dir": str(model_dir),
        "seed": args.seed,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
