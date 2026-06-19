import argparse
import math
import random
import tarfile
import time
import urllib.request
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from torch import nn


PTB_URL = "http://www.fit.vutbr.cz/~imikolov/rnnlm/simple-examples.tgz"
PTB_FILENAMES = {
    "train": "ptb.train.txt",
    "valid": "ptb.valid.txt",
    "test": "ptb.test.txt",
}


PRESETS = {
    # Fast sanity check. This is useful for verifying the pipeline, not for the
    # final perplexity target.
    "demo": {
        "embedding_size": 128,
        "hidden_size": 128,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 20,
        "bptt": 35,
        "learning_rate": 10.0,
        "max_epoch": 4,
        "epochs": 8,
        "lr_decay": 0.5,
        "clip": 0.25,
        "tie_weights": True,
    },
    # Close to the classic small PTB LSTM baseline.
    "small": {
        "embedding_size": 200,
        "hidden_size": 200,
        "num_layers": 2,
        "dropout": 0.2,
        "batch_size": 20,
        "bptt": 35,
        "learning_rate": 20.0,
        "max_epoch": 4,
        "epochs": 13,
        "lr_decay": 0.5,
        "clip": 0.25,
        "tie_weights": True,
    },
    # Recommended preset for the report requirement. With enough epochs it is
    # expected to reach validation/test PPL below 80 on PTB.
    "medium": {
        "embedding_size": 650,
        "hidden_size": 650,
        "num_layers": 2,
        "dropout": 0.5,
        "batch_size": 20,
        "bptt": 35,
        "learning_rate": 20.0,
        "max_epoch": 6,
        "epochs": 39,
        "lr_decay": 0.8,
        "clip": 0.25,
        "tie_weights": True,
    },
    # Continue from a near-target medium checkpoint with a smaller learning
    # rate. This is intended for squeezing validation PPL below 80.
    "finetune": {
        "embedding_size": 650,
        "hidden_size": 650,
        "num_layers": 2,
        "dropout": 0.5,
        "batch_size": 20,
        "bptt": 35,
        "learning_rate": 1.0,
        "max_epoch": 1,
        "epochs": 8,
        "lr_decay": 0.8,
        "clip": 0.25,
        "tie_weights": True,
    },
}


def safe_exp(value):
    return float("inf") if value > 709 else math.exp(value)


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def find_ptb_dir(data_dir):
    data_dir = Path(data_dir)
    candidates = [
        data_dir,
        data_dir / "simple-examples" / "data",
        data_dir.parent / "simple-examples" / "data",
    ]
    for candidate in candidates:
        if all((candidate / name).exists() for name in PTB_FILENAMES.values()):
            return candidate
    return None


def safe_extract_tar(tar, target_dir):
    target_dir = Path(target_dir).resolve()
    for member in tar.getmembers():
        member_path = (target_dir / member.name).resolve()
        if target_dir not in member_path.parents and member_path != target_dir:
            raise RuntimeError(f"Unsafe archive member: {member.name}")
    tar.extractall(target_dir)


def download_ptb(data_dir):
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    archive_path = data_dir / "simple-examples.tgz"

    if not archive_path.exists():
        print(f"Downloading PTB from {PTB_URL}")
        urllib.request.urlretrieve(PTB_URL, archive_path)

    print(f"Extracting {archive_path}")
    with tarfile.open(archive_path, "r:gz") as tar:
        safe_extract_tar(tar, data_dir)

    ptb_dir = find_ptb_dir(data_dir)
    if ptb_dir is None:
        raise FileNotFoundError("PTB files were not found after extraction.")
    return ptb_dir


def read_words(path):
    with Path(path).open("r", encoding="utf-8") as f:
        return f.read().replace("\n", " <eos> ").split()


def build_vocab(train_path):
    counter = Counter(read_words(train_path))
    words = sorted(counter.items(), key=lambda item: (-item[1], item[0]))
    word_to_id = {word: idx for idx, (word, _) in enumerate(words)}
    id_to_word = [word for word, _ in words]
    return word_to_id, id_to_word


def file_to_ids(path, word_to_id):
    return [word_to_id[word] for word in read_words(path)]


def load_corpus(data_dir, download=False):
    ptb_dir = find_ptb_dir(data_dir)
    if ptb_dir is None and download:
        ptb_dir = download_ptb(data_dir)
    if ptb_dir is None:
        expected = ", ".join(PTB_FILENAMES.values())
        raise FileNotFoundError(
            f"Could not find PTB files ({expected}) under {Path(data_dir).resolve()}. "
            "Use --download or place the files in --data-dir."
        )

    word_to_id, id_to_word = build_vocab(ptb_dir / PTB_FILENAMES["train"])
    corpus = {
        split: torch.tensor(file_to_ids(ptb_dir / filename, word_to_id), dtype=torch.long)
        for split, filename in PTB_FILENAMES.items()
    }
    return corpus, word_to_id, id_to_word, ptb_dir


def batchify(data, batch_size, device):
    nbatch = data.size(0) // batch_size
    data = data.narrow(0, 0, nbatch * batch_size)
    data = data.view(batch_size, -1).t().contiguous()
    return data.to(device)


def get_batch(source, start_index, bptt):
    seq_len = min(bptt, source.size(0) - 1 - start_index)
    data = source[start_index : start_index + seq_len]
    target = source[start_index + 1 : start_index + 1 + seq_len].reshape(-1)
    return data, target


def detach_hidden(hidden):
    return tuple(item.detach() for item in hidden)


class LockedDropout(nn.Module):
    def __init__(self, dropout):
        super().__init__()
        self.dropout = dropout

    def forward(self, inputs):
        if not self.training or self.dropout <= 0:
            return inputs
        mask = inputs.new_empty(1, inputs.size(1), inputs.size(2)).bernoulli_(1 - self.dropout)
        mask = mask.div_(1 - self.dropout)
        return inputs * mask


class LSTMLanguageModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        embedding_size,
        hidden_size,
        num_layers,
        dropout,
        tie_weights,
    ):
        super().__init__()
        if tie_weights and embedding_size != hidden_size:
            raise ValueError("tie_weights requires embedding_size == hidden_size")

        self.drop = LockedDropout(dropout)
        self.encoder = nn.Embedding(vocab_size, embedding_size)
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.rnn = nn.LSTM(
            embedding_size,
            hidden_size,
            num_layers,
            dropout=lstm_dropout,
        )
        self.decoder = nn.Linear(hidden_size, vocab_size)
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        if tie_weights:
            self.decoder.weight = self.encoder.weight

        self.init_weights()

    def init_weights(self):
        init_range = 0.1
        nn.init.uniform_(self.encoder.weight, -init_range, init_range)
        nn.init.zeros_(self.decoder.bias)
        nn.init.uniform_(self.decoder.weight, -init_range, init_range)

    def forward(self, inputs, hidden, return_features=False):
        embeddings = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(embeddings, hidden)
        output = self.drop(output)
        features = output.reshape(output.size(0) * output.size(1), output.size(2))
        decoded = self.decoder(features)
        if return_features:
            return decoded, hidden, features
        return decoded, hidden

    def init_hidden(self, batch_size, device):
        weight = next(self.parameters())
        return (
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size, device=device),
            weight.new_zeros(self.num_layers, batch_size, self.hidden_size, device=device),
        )


def run_epoch(model, data_source, criterion, args, optimizer=None):
    is_train = optimizer is not None
    model.train(is_train)
    total_loss = 0.0
    total_tokens = 0
    start_time = time.time()
    hidden = model.init_hidden(args.batch_size, args.device)

    for batch, start_index in enumerate(range(0, data_source.size(0) - 1, args.bptt), 1):
        inputs, targets = get_batch(data_source, start_index, args.bptt)
        hidden = detach_hidden(hidden)

        if is_train:
            optimizer.zero_grad(set_to_none=True)

        output, hidden = model(inputs, hidden)
        loss = criterion(output, targets)

        if is_train:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            optimizer.step()

        tokens = targets.numel()
        total_loss += loss.item() * tokens
        total_tokens += tokens

        if is_train and args.log_interval > 0 and batch % args.log_interval == 0:
            elapsed = time.time() - start_time
            cur_loss = total_loss / total_tokens
            wps = total_tokens / max(elapsed, 1e-9)
            print(
                f"  batch {batch:5d} | ppl {safe_exp(cur_loss):8.2f} | "
                f"loss {cur_loss:5.2f} | {wps:8.0f} tokens/s"
            )

    return total_loss / total_tokens


def run_ensemble(models, data_source, args):
    for model in models:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden_states = [model.init_hidden(args.batch_size, args.device) for model in models]

    with torch.no_grad():
        for start_index in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, start_index, args.bptt)
            probabilities = []
            for model_index, model in enumerate(models):
                output, hidden_states[model_index] = model(inputs, hidden_states[model_index])
                hidden_states[model_index] = detach_hidden(hidden_states[model_index])
                probabilities.append(torch.softmax(output, dim=1))

            mean_prob = torch.stack(probabilities, dim=0).mean(dim=0)
            token_prob = mean_prob.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            total_loss += -torch.log(token_prob).sum().item()
            total_tokens += targets.numel()

    return total_loss / total_tokens


def update_cache(cache_keys, cache_values, features, targets, cache_size):
    if cache_size <= 0:
        return cache_keys, cache_values
    features = features.detach()
    targets = targets.detach()
    if cache_keys is None:
        cache_keys = features[-cache_size:]
        cache_values = targets[-cache_size:]
    else:
        cache_keys = torch.cat([cache_keys, features], dim=0)[-cache_size:]
        cache_values = torch.cat([cache_values, targets], dim=0)[-cache_size:]
    return cache_keys, cache_values


def apply_neural_cache(probabilities, features, cache_keys, cache_values, args):
    if cache_keys is None or cache_keys.size(0) == 0 or args.cache_lambda <= 0:
        return probabilities

    similarity = torch.matmul(features, cache_keys.t()) * args.cache_theta
    cache_weights = torch.softmax(similarity, dim=1)
    cache_probabilities = probabilities.new_zeros(probabilities.size())
    expanded_values = cache_values.unsqueeze(0).expand(features.size(0), -1)
    cache_probabilities.scatter_add_(1, expanded_values, cache_weights)
    return (1.0 - args.cache_lambda) * probabilities + args.cache_lambda * cache_probabilities


def run_cached_eval(model, data_source, args):
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden = model.init_hidden(args.batch_size, args.device)
    cache_keys = None
    cache_values = None

    with torch.no_grad():
        for start_index in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, start_index, args.bptt)
            output, hidden, features = model(inputs, hidden, return_features=True)
            hidden = detach_hidden(hidden)
            probabilities = torch.softmax(output, dim=1)
            probabilities = apply_neural_cache(probabilities, features, cache_keys, cache_values, args)
            token_prob = probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            total_loss += -torch.log(token_prob).sum().item()
            total_tokens += targets.numel()
            cache_keys, cache_values = update_cache(
                cache_keys,
                cache_values,
                features,
                targets,
                args.cache_size,
            )

    return total_loss / total_tokens


def run_cached_ensemble(models, data_source, args):
    for model in models:
        model.eval()
    total_loss = 0.0
    total_tokens = 0
    hidden_states = [model.init_hidden(args.batch_size, args.device) for model in models]
    cache_keys = None
    cache_values = None

    with torch.no_grad():
        for start_index in range(0, data_source.size(0) - 1, args.bptt):
            inputs, targets = get_batch(data_source, start_index, args.bptt)
            probabilities = []
            feature_list = []
            for model_index, model in enumerate(models):
                output, hidden_states[model_index], features = model(
                    inputs,
                    hidden_states[model_index],
                    return_features=True,
                )
                hidden_states[model_index] = detach_hidden(hidden_states[model_index])
                probabilities.append(torch.softmax(output, dim=1))
                feature_list.append(features)

            mean_probabilities = torch.stack(probabilities, dim=0).mean(dim=0)
            mean_features = torch.stack(feature_list, dim=0).mean(dim=0)
            mean_probabilities = apply_neural_cache(
                mean_probabilities,
                mean_features,
                cache_keys,
                cache_values,
                args,
            )
            token_prob = mean_probabilities.gather(1, targets.unsqueeze(1)).squeeze(1).clamp_min(1e-12)
            total_loss += -torch.log(token_prob).sum().item()
            total_tokens += targets.numel()
            cache_keys, cache_values = update_cache(
                cache_keys,
                cache_values,
                mean_features,
                targets,
                args.cache_size,
            )

    return total_loss / total_tokens


def save_checkpoint(path, model, args, word_to_id, id_to_word, valid_ppl):
    checkpoint = {
        "model_state": model.state_dict(),
        "config": {
            "embedding_size": args.embedding_size,
            "hidden_size": args.hidden_size,
            "num_layers": args.num_layers,
            "dropout": args.dropout,
            "tie_weights": args.tie_weights,
        },
        "word_to_id": word_to_id,
        "id_to_word": id_to_word,
        "valid_ppl": valid_ppl,
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, path)


def check_checkpoint_config(checkpoint, args):
    checkpoint_config = checkpoint.get("config", {})
    expected = {
        "embedding_size": args.embedding_size,
        "hidden_size": args.hidden_size,
        "num_layers": args.num_layers,
        "tie_weights": args.tie_weights,
    }
    mismatched = [
        f"{key}: checkpoint={checkpoint_config.get(key)} current={value}"
        for key, value in expected.items()
        if checkpoint_config.get(key) != value
    ]
    if mismatched:
        details = "; ".join(mismatched)
        raise ValueError(f"Checkpoint config does not match current model: {details}")


def load_best_model(path, model, device):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model_state"])
    return checkpoint


def make_optimizer(model, args):
    if args.optimizer == "sgd":
        return torch.optim.SGD(model.parameters(), lr=args.learning_rate)
    if args.optimizer == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
    raise ValueError(f"Unsupported optimizer: {args.optimizer}")


def parse_args():
    parser = argparse.ArgumentParser(description="Train an LSTM language model on PTB.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/ptb"))
    parser.add_argument("--download", action="store_true", help="Download PTB if it is missing.")
    parser.add_argument(
        "--prepare-only",
        action="store_true",
        help="Only download/load PTB and build the vocabulary, then exit.",
    )
    parser.add_argument("--preset", choices=sorted(PRESETS), default="medium")
    parser.add_argument("--embedding-size", type=int)
    parser.add_argument("--hidden-size", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--dropout", type=float)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--bptt", type=int)
    parser.add_argument("--learning-rate", type=float)
    parser.add_argument("--max-epoch", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--lr-decay", type=float)
    parser.add_argument("--clip", type=float)
    parser.add_argument("--optimizer", choices=["sgd", "adamw"], default="sgd")
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--tie-weights", action=argparse.BooleanOptionalAction)
    parser.add_argument("--seed", type=int, default=1111)
    parser.add_argument("--cuda", action="store_true", help="Use CUDA when available.")
    parser.add_argument("--save", type=Path, default=Path("checkpoints/ptb_lstm.pt"))
    parser.add_argument(
        "--resume",
        type=Path,
        help="Continue training from an existing checkpoint.",
    )
    parser.add_argument(
        "--eval-only",
        type=Path,
        help="Evaluate a checkpoint on validation and test sets, then exit.",
    )
    parser.add_argument(
        "--ensemble-eval",
        nargs="+",
        type=Path,
        help="Evaluate checkpoints by averaging their predicted probabilities.",
    )
    parser.add_argument(
        "--early-stop",
        type=int,
        default=0,
        help="Stop after this many epochs without validation improvement. 0 disables it.",
    )
    parser.add_argument(
        "--cache-size",
        type=int,
        default=0,
        help="Neural cache window size used during evaluation. 0 disables cache.",
    )
    parser.add_argument(
        "--cache-lambda",
        type=float,
        default=0.0,
        help="Interpolation weight for neural cache probabilities.",
    )
    parser.add_argument(
        "--cache-theta",
        type=float,
        default=0.2,
        help="Sharpness of neural cache hidden-state similarities.",
    )
    parser.add_argument("--log-interval", type=int, default=200)
    args = parser.parse_args()

    defaults = PRESETS[args.preset]
    for key, value in defaults.items():
        if getattr(args, key) is None:
            setattr(args, key, value)

    if args.cuda and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
    args.device = torch.device("cuda" if args.cuda and torch.cuda.is_available() else "cpu")
    return args


def main():
    args = parse_args()
    set_seed(args.seed)

    corpus, word_to_id, id_to_word, ptb_dir = load_corpus(args.data_dir, args.download)
    vocab_size = len(word_to_id)
    print(f"PTB directory: {ptb_dir}")
    print(f"Vocabulary size: {vocab_size}")
    if args.prepare_only:
        for split, data in corpus.items():
            print(f"{split:>5} tokens: {data.numel()}")
        print("PTB data is ready.")
        return

    print(f"Device: {args.device}")
    print(f"Preset: {args.preset}")
    print(f"Epochs: {args.epochs} | lr: {args.learning_rate} | dropout: {args.dropout}")

    train_data = batchify(corpus["train"], args.batch_size, args.device)
    valid_data = batchify(corpus["valid"], args.batch_size, args.device)
    test_data = batchify(corpus["test"], args.batch_size, args.device)

    if args.cache_size > 0 and args.batch_size != 1:
        raise ValueError("Neural cache evaluation requires --batch-size 1.")

    model = LSTMLanguageModel(
        vocab_size=vocab_size,
        embedding_size=args.embedding_size,
        hidden_size=args.hidden_size,
        num_layers=args.num_layers,
        dropout=args.dropout,
        tie_weights=args.tie_weights,
    ).to(args.device)
    criterion = nn.CrossEntropyLoss()

    if args.ensemble_eval:
        models = []
        saved_valid_ppls = []
        for checkpoint_path in args.ensemble_eval:
            ensemble_model = LSTMLanguageModel(
                vocab_size=vocab_size,
                embedding_size=args.embedding_size,
                hidden_size=args.hidden_size,
                num_layers=args.num_layers,
                dropout=args.dropout,
                tie_weights=args.tie_weights,
            ).to(args.device)
            checkpoint = load_best_model(checkpoint_path, ensemble_model, args.device)
            check_checkpoint_config(checkpoint, args)
            models.append(ensemble_model)
            saved_valid_ppls.append(checkpoint.get("valid_ppl", float("nan")))

        eval_fn = run_cached_ensemble if args.cache_size > 0 and args.cache_lambda > 0 else run_ensemble
        valid_loss = eval_fn(models, valid_data, args)
        test_loss = eval_fn(models, test_data, args)
        print("Ensemble checkpoints:")
        for checkpoint_path, saved_valid_ppl in zip(args.ensemble_eval, saved_valid_ppls):
            print(f"  {checkpoint_path} | saved valid ppl {saved_valid_ppl:.3f}")
        label = "Cached ensemble" if eval_fn is run_cached_ensemble else "Ensemble"
        print(f"{label} valid ppl: {safe_exp(valid_loss):.3f}")
        print(f"{label} test ppl: {safe_exp(test_loss):.3f}")
        return

    if args.eval_only:
        checkpoint = load_best_model(args.eval_only, model, args.device)
        check_checkpoint_config(checkpoint, args)
        if args.cache_size > 0 and args.cache_lambda > 0:
            valid_loss = run_cached_eval(model, valid_data, args)
            test_loss = run_cached_eval(model, test_data, args)
        else:
            valid_loss = run_epoch(model, valid_data, criterion, args)
            test_loss = run_epoch(model, test_data, criterion, args)
        print(f"Checkpoint: {args.eval_only}")
        print(f"Saved valid ppl: {checkpoint.get('valid_ppl', float('nan')):.3f}")
        print(f"Eval valid ppl: {safe_exp(valid_loss):.3f}")
        print(f"Eval test ppl: {safe_exp(test_loss):.3f}")
        return

    optimizer = make_optimizer(model, args)

    best_valid_ppl = float("inf")
    best_checkpoint_path = None
    epochs_without_improvement = 0
    if args.resume:
        checkpoint = load_best_model(args.resume, model, args.device)
        check_checkpoint_config(checkpoint, args)
        best_valid_ppl = checkpoint.get("valid_ppl", best_valid_ppl)
        best_checkpoint_path = args.resume
        print(f"Resumed model from {args.resume}")
        print(f"Checkpoint best valid ppl: {best_valid_ppl:.3f}")

    for epoch in range(1, args.epochs + 1):
        lr = args.learning_rate * (args.lr_decay ** max(epoch - args.max_epoch, 0))
        for group in optimizer.param_groups:
            group["lr"] = lr

        print(f"\nEpoch {epoch:02d}/{args.epochs} | lr {lr:.4f}")
        train_loss = run_epoch(model, train_data, criterion, args, optimizer)
        valid_loss = run_epoch(model, valid_data, criterion, args)
        train_ppl = safe_exp(train_loss)
        valid_ppl = safe_exp(valid_loss)
        print(f"Epoch {epoch:02d} | train ppl {train_ppl:.3f} | valid ppl {valid_ppl:.3f}")

        if valid_ppl < best_valid_ppl:
            best_valid_ppl = valid_ppl
            best_checkpoint_path = args.save
            epochs_without_improvement = 0
            save_checkpoint(args.save, model, args, word_to_id, id_to_word, valid_ppl)
            print(f"  saved best model to {args.save}")
        else:
            epochs_without_improvement += 1

        if args.early_stop and epochs_without_improvement >= args.early_stop:
            print(f"Early stopping: no validation improvement for {args.early_stop} epochs.")
            break

    if best_checkpoint_path and best_checkpoint_path.exists():
        checkpoint = load_best_model(best_checkpoint_path, model, args.device)
        best_valid_ppl = checkpoint["valid_ppl"]

    test_loss = run_epoch(model, test_data, criterion, args)
    test_ppl = safe_exp(test_loss)
    print(f"\nBest valid ppl: {best_valid_ppl:.3f}")
    print(f"Test ppl: {test_ppl:.3f}")


if __name__ == "__main__":
    main()
