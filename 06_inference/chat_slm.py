#!/usr/bin/env python3
"""
Interactive inference for the London Historical SLM checkpoints (.pt)
Loads the latest checkpoint from a directory (default: 09_models/checkpoints/slm_full)
and offers both single-prompt and interactive chat modes.
"""

import os
import sys
import argparse
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional, List, Tuple

# Reduce noisy logs/warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.append(str(project_root))

try:
    import torch
    from transformers import AutoTokenizer
except ImportError as e:
    print(f"Missing required dependencies: {e}")
    print("Please install requirements: pip install -r requirements.txt")
    sys.exit(1)


def _infer_arch_from_state_dict(state_dict, tokenizer_vocab_size: int, fallback_n_head: int = 12):
    """Infer model architecture from a SimpleGPT state dict.

    Returns: (n_layer, n_embd, n_head, block_size, vocab_size)
    """
    # Normalize keys (strip _orig_mod.)
    keys = [k.replace('_orig_mod.', '') for k in state_dict.keys()]

    # n_layer
    import re
    layer_nums = []
    for k in keys:
        m = re.search(r"transformer\.h\.(\d+)\.", k)
        if m:
            layer_nums.append(int(m.group(1)))
    n_layer = max(layer_nums) + 1 if layer_nums else 8

    # n_embd & vocab_size from wte
    wte_key = 'transformer.wte.weight'
    wte_key_alt = '_orig_mod.transformer.wte.weight'
    if wte_key in state_dict:
        wte = state_dict[wte_key]
    elif wte_key_alt in state_dict:
        wte = state_dict[wte_key_alt]
    else:
        wte = None
    if wte is not None and wte.ndim == 2:
        vocab_size = wte.shape[0]
        n_embd = wte.shape[1]
    else:
        vocab_size = tokenizer_vocab_size
        n_embd = 512

    # block_size from wpe
    wpe_key = 'transformer.wpe.weight'
    wpe_key_alt = '_orig_mod.transformer.wpe.weight'
    if wpe_key in state_dict:
        wpe = state_dict[wpe_key]
    elif wpe_key_alt in state_dict:
        wpe = state_dict[wpe_key_alt]
    else:
        wpe = None
    block_size = wpe.shape[0] if (wpe is not None and wpe.ndim == 2) else 256

    # n_head (cannot be inferred directly from weights) – use training config default
    # Prefer the project config if available
    try:
        from config import config as global_config
        n_head = int(global_config.slm_config.get("n_head", fallback_n_head))
    except Exception:
        n_head = fallback_n_head

    return n_layer, n_embd, n_head, block_size, vocab_size


def load_simple_model(checkpoint_path: Path, tokenizer_dir: Path, device: torch.device):
    """Load SimpleGPT model from a training checkpoint (.pt) and tokenizer."""
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_dir))
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load checkpoint
    ckpt = torch.load(checkpoint_path, map_location=device)
    if isinstance(ckpt, dict) and 'model' in ckpt:
        state_dict = ckpt['model']
    elif isinstance(ckpt, dict):
        state_dict = ckpt
    else:
        raise RuntimeError("Unsupported checkpoint format: expected dict with 'model' key or raw state dict")

    # Build SimpleGPT
    sys.path.append(str(project_root / "04_training"))
    from train_model_slm import SimpleGPT, SimpleGPTConfig

    n_layer, n_embd, n_head, block_size, vocab_size = _infer_arch_from_state_dict(
        state_dict, tokenizer_vocab_size=tokenizer.vocab_size
    )

    cfg = SimpleGPTConfig(
        n_layer=n_layer,
        n_head=n_head,
        n_embd=n_embd,
        block_size=block_size,
        bias=False,
        vocab_size=vocab_size,
        dropout=0.1,
    )
    model = SimpleGPT(cfg)

    # Strip _orig_mod. prefix (from torch.compile) if present
    cleaned = {}
    prefix = '_orig_mod.'
    for k, v in state_dict.items():
        if k.startswith(prefix):
            cleaned_key = k[len(prefix):]
        else:
            cleaned_key = k
        cleaned[cleaned_key] = v
    model.load_state_dict(cleaned)
    model.to(device)
    model.eval()

    return model, tokenizer, cfg


@torch.no_grad()
def generate_with_simple(model, tokenizer, prompt: str, max_new_tokens: int = 120,
                         temperature: float = 0.8, top_p: float = 0.9, device: torch.device = torch.device('cpu')) -> str:
    """Generate continuation for a single string prompt (no chat memory)."""
    enc = tokenizer(prompt, return_tensors='pt', add_special_tokens=False)
    input_ids = enc['input_ids'].to(device)
    return generate_with_simple_ids(model, tokenizer, input_ids, max_new_tokens, temperature, top_p, device)


@torch.no_grad()
def generate_with_simple_ids(model, tokenizer, input_ids: torch.Tensor, max_new_tokens: int = 120,
                             temperature: float = 0.8, top_p: float = 0.9, device: torch.device = torch.device('cpu')) -> str:
    """Generate given already-tokenized input (used for chat with memory)."""
    generated = input_ids.clone()
    eos_id = tokenizer.eos_token_id

    for _ in range(max_new_tokens):
        # Respect block size
        block_size = getattr(model.config, 'block_size', 256)
        if generated.size(1) > block_size:
            generated = generated[:, -block_size:]

        logits, _ = model(generated)
        logits = logits[:, -1, :] / max(temperature, 1e-6)
        probs = torch.softmax(logits, dim=-1)

        if top_p < 1.0:
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=-1)
            cutoff = cumulative > top_p
            cutoff[..., 0] = False
            sorted_probs[cutoff] = 0
            sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
            next_token_sorted = torch.multinomial(sorted_probs, num_samples=1)
            next_token = sorted_indices.gather(-1, next_token_sorted)
        else:
            next_token = torch.multinomial(probs, num_samples=1)

        generated = torch.cat([generated, next_token], dim=1)

        if eos_id is not None and next_token.item() == eos_id:
            break

    # Decode only the new tokens
    new_tokens = generated[0, input_ids.shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


# --------------------------- Chat with memory ---------------------------

ROLE_USER = "User"
ROLE_ASSISTANT = "Assistant"
ROLE_SYSTEM = "System"


@dataclass
class ChatSession:
    tokenizer: any
    device: torch.device
    system_prompt: str = "You are a helpful assistant specialized in 1500-1850 London history."
    messages: List[Tuple[str, str]] = field(default_factory=list)  # list of (role, content)

    def add_user(self, content: str):
        self.messages.append((ROLE_USER, content))

    def add_assistant(self, content: str):
        self.messages.append((ROLE_ASSISTANT, content))

    def clear(self):
        self.messages.clear()

    def build_prompt_text(self) -> str:
        parts = []
        if self.system_prompt:
            parts.append(f"{ROLE_SYSTEM}: {self.system_prompt}\n\n")
        for role, content in self.messages:
            parts.append(f"{role}: {content}\n\n")
        # End with assistant cue
        parts.append(f"{ROLE_ASSISTANT}: ")
        return "".join(parts)

    def build_input_ids_clipped(self, model_block_size: int) -> torch.Tensor:
        """Tokenize the chat context and clip to the model block size from the end."""
        text = self.build_prompt_text()
        enc = self.tokenizer(text, return_tensors='pt', add_special_tokens=False)
        input_ids = enc['input_ids']
        # Clip to block_size
        if input_ids.shape[1] > model_block_size:
            input_ids = input_ids[:, -model_block_size:]
        return input_ids.to(self.device)


def pick_latest_checkpoint(model_dir: Path) -> Optional[Path]:
    candidates = list(model_dir.glob("checkpoint-*.pt"))
    if not candidates:
        return None
    # Sort by numeric step in filename
    def step_num(p: Path) -> int:
        try:
            return int(p.stem.split('-')[1])
        except Exception:
            return -1
    return sorted(candidates, key=step_num)[-1]


def main():
    parser = argparse.ArgumentParser(description="Chat with the London Historical SLM (local checkpoints)")
    parser.add_argument("--model_dir", type=str, default="09_models/checkpoints/slm_full",
                        help="Directory containing checkpoint-*.pt files")
    parser.add_argument("--checkpoint", type=str, default=None,
                        help="Path to a specific checkpoint .pt file (overrides --model_dir)")
    parser.add_argument("--tokenizer_dir", type=str, default="09_models/tokenizers/london_historical_tokenizer",
                        help="Directory of the tokenizer")
    parser.add_argument("--device", type=str, choices=["auto", "gpu", "cpu"], default="auto",
                        help="Device to run on")
    parser.add_argument("--prompt", type=str, default=None,
                        help="Single prompt to run once and exit")
    parser.add_argument("--interactive", action="store_true",
                        help="Enter interactive chat mode (with memory)")
    parser.add_argument("--max_new_tokens", type=int, default=120)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.9)

    args = parser.parse_args()

    # Device selection
    if args.device == "gpu" or (args.device == "auto" and torch.cuda.is_available()):
        device = torch.device("cuda")
        print("🚀 Using GPU")
    else:
        device = torch.device("cpu")
        if args.device == "gpu" and not torch.cuda.is_available():
            print("⚠️  GPU requested but not available; using CPU")
        else:
            print("🖥️  Using CPU")

    model_dir = Path(args.model_dir)
    tokenizer_dir = Path(args.tokenizer_dir)

    # Resolve checkpoint
    if args.checkpoint:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = pick_latest_checkpoint(model_dir)

    if not ckpt_path or not ckpt_path.exists():
        print(f"❌ No checkpoint found. Looked for checkpoint-*.pt in {model_dir} or explicit --checkpoint path.")
        return 1

    print(f"📦 Loading checkpoint: {ckpt_path}")
    print(f"🔤 Tokenizer: {tokenizer_dir}")

    try:
        model, tokenizer, cfg = load_simple_model(ckpt_path, tokenizer_dir, device)
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        return 1

    def run_one(prompt: str):
        out = generate_with_simple(
            model, tokenizer, prompt,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            device=device,
        )
        print("\n📖 Generated:\n" + ("-" * 50))
        print(out)
        print("-" * 50)

    # Single prompt
    if args.prompt and not args.interactive:
        run_one(args.prompt)
        return 0

    # Interactive chat with memory
    block_size = getattr(cfg, 'block_size', 256)
    chat = ChatSession(tokenizer=tokenizer, device=device)
    log_dir = project_root / "06_inference" / "chat_logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    session_started = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print("\n🎭 Interactive mode — type '/help' for commands. Continuous chat is enabled.\n")

    def save_transcript(path: Optional[Path] = None):
        ts = datetime.now().strftime('%Y%m%d-%H%M%S')
        out_path = path if path else (log_dir / f"chat-{ts}.txt")
        lines = [f"Session start: {session_started}", f"System: {chat.system_prompt}", ""]
        for role, content in chat.messages:
            lines.append(f"{role}: {content}")
        out_path.write_text("\n".join(lines), encoding='utf-8')
        print(f"💾 Saved transcript to: {out_path}")

    while True:
        try:
            user_in = input("📝 You> ").strip()
            if not user_in:
                continue

            # Commands
            if user_in.startswith('/'):
                cmd, *rest = user_in.split(' ', 1)
                arg = (rest[0] if rest else '').strip()
                if cmd in {'/quit', '/exit'}:
                    break
                elif cmd == '/help':
                    print("\nCommands:\n  /help              Show commands\n  /quit or /exit     Quit chat\n  /clear             Clear conversation history\n  /sys <text>        Set system prompt\n  /save [path]       Save transcript (optional path)\n  /config            Show current settings\n  /temp <val>        Set temperature (e.g., 0.7)\n  /top_p <val>       Set top_p (e.g., 0.9)\n  /maxnew <int>      Set max_new_tokens (e.g., 120)\n  /more              Ask the model to continue last answer\n")
                elif cmd == '/clear':
                    chat.clear()
                    print("✅ History cleared")
                elif cmd == '/sys':
                    if arg:
                        chat.system_prompt = arg
                        print("✅ System prompt updated")
                    else:
                        print("Usage: /sys <text>")
                elif cmd == '/config':
                    print(f"\nSettings: max_new_tokens={args.max_new_tokens}, temperature={args.temperature}, top_p={args.top_p}, block_size={block_size}")
                    print(f"System: {chat.system_prompt}\n")
                elif cmd == '/save':
                    out_path = Path(arg) if arg else None
                    save_transcript(out_path)
                elif cmd == '/temp':
                    try:
                        args.temperature = float(arg)
                        print("✅ temperature set")
                    except Exception:
                        print("Usage: /temp <float>")
                elif cmd == '/top_p':
                    try:
                        args.top_p = float(arg)
                        print("✅ top_p set")
                    except Exception:
                        print("Usage: /top_p <float>")
                elif cmd == '/maxnew':
                    try:
                        args.max_new_tokens = int(arg)
                        print("✅ max_new_tokens set")
                    except Exception:
                        print("Usage: /maxnew <int>")
                elif cmd == '/more':
                    # Ask model to continue given current context
                    input_ids = chat.build_input_ids_clipped(block_size)
                    reply = generate_with_simple_ids(
                        model, tokenizer, input_ids,
                        max_new_tokens=args.max_new_tokens,
                        temperature=args.temperature,
                        top_p=args.top_p,
                        device=device,
                    )
                    chat.add_assistant(reply)
                    print("\n🤖 Assistant>\n" + ("-" * 50))
                    print(reply)
                    print("-" * 50)
                else:
                    print("Unknown command. Type /help")
                continue

            # Regular user message
            chat.add_user(user_in)
            input_ids = chat.build_input_ids_clipped(block_size)
            reply = generate_with_simple_ids(
                model, tokenizer, input_ids,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                device=device,
            )
            chat.add_assistant(reply)
            print("\n🤖 Assistant>\n" + ("-" * 50))
            print(reply)
            print("-" * 50)

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break
        except Exception as e:
            print(f"❌ Error: {e}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
