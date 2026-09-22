"""Reproducible training and held-out feature export."""

from pathlib import Path
import copy
import hashlib
import json
import math
import os
import random
import numpy as np
from .data import subject_split, normalization
from .integer import IntegerHead


def train(data_path, out_dir, config, seed=7, explicit_split=None):
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset
    from torch.ao.quantization import disable_observer, enable_observer
    from .model import TinyHAR, model_profile

    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(int(config.get("threads", 2)))
    device = torch.device(config.get("device", "cpu"))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with np.load(data_path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    x, y, subjects = data["x"], data["y"], data["subject"].astype(str)
    if x.ndim != 3 or not np.isfinite(x).all():
        raise ValueError("expected finite [N,C,T] windows")
    split = subject_split(subjects, seed, explicit_split)
    indices = {role: np.flatnonzero(np.isin(subjects, users)) for role, users in split.items()}
    if config.get("id_placement"):
        for role in ("train", "val", "cal"):
            ix = indices[role]
            indices[role] = ix[data["placement"][ix] == config["id_placement"]]
            if not len(indices[role]):
                raise ValueError(f"no ID placement windows in {role}")
    classes = len(data["classes"])
    if set(y[indices["train"]]) != set(range(classes)):
        raise ValueError("training subjects do not contain every class; specify a valid split")
    mean, std = normalization(x[indices["train"]])
    x = ((x - mean) / std).astype(np.float32)[:, None]
    np.savez(out / "normalization.npz", mean=mean, std=std)
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")
    model = TinyHAR(classes, int(config.get("feature_dim", 128))).to(device)
    loader = DataLoader(
        TensorDataset(torch.from_numpy(x[indices["train"]]), torch.from_numpy(y[indices["train"]])),
        batch_size=int(config.get("batch_size", 128)),
        shuffle=True,
        num_workers=0,
        generator=torch.Generator().manual_seed(seed),
    )
    epochs, patience = int(config.get("epochs", 80)), int(config.get("patience", 12))
    if min(epochs, patience) < 1:
        raise ValueError("epochs/patience must be positive")
    lr, floor = float(config.get("lr", 0.001)), float(config.get("lr_floor", 1e-5))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    weights = None
    if config.get("class_balanced", False):
        counts = np.bincount(y[indices["train"]], minlength=classes)
        weights = torch.tensor(
            len(indices["train"]) / (classes * counts), dtype=torch.float32, device=device
        )
    ce = nn.CrossEntropyLoss(weight=weights)
    history, best_loss, stale, best = [], float("inf"), 0, None
    for epoch in range(epochs):
        model.train()
        model.apply(enable_observer if epoch < max(1, int(0.75 * epochs)) else disable_observer)
        warmup = min(int(config.get("warmup_epochs", 5)), epochs)
        rate = (
            lr * (epoch + 1) / warmup
            if epoch < warmup
            else floor
            + (lr - floor)
            * 0.5
            * (1 + math.cos(math.pi * (epoch - warmup) / max(1, epochs - warmup - 1)))
        )
        for group in optimizer.param_groups:
            group["lr"] = rate
        total, seen = 0.0, 0
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            if config.get("augment", True):
                xb = xb + 0.01 * torch.randn_like(xb)
                xb = xb * (0.95 + 0.1 * torch.rand(len(xb), 1, 1, 1, device=device))
                shifts = torch.randint(-3, 4, (len(xb),), device=device)
                xb = torch.stack([torch.roll(v, int(s), dims=-1) for v, s in zip(xb, shifts)])
                xb *= torch.randint(0, 2, (len(xb), 1, xb.shape[2], 1), device=device) * 2 - 1
            optimizer.zero_grad(set_to_none=True)
            logits, z, margin = model(xb, masked=True)
            full = model.head(z)
            target = torch.topk(full.detach(), 2, dim=-1).values.diff(dim=-1).abs().squeeze(-1)
            loss = (
                0.5 * (ce(logits, yb) + ce(full, yb))
                + 0.05 * nn.functional.smooth_l1_loss(margin, target)
                + 0.2 * ce(model.dropout_head(z.detach()), yb)
            )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.detach()) * len(xb)
            seen += len(xb)
        model.eval()
        model.apply(disable_observer)
        val_loss = 0.0
        with torch.no_grad():
            ix = indices["val"]
            for start in range(0, len(ix), 128):
                take = ix[start : start + 128]
                vl, _, _ = model(torch.from_numpy(x[take]).to(device))
                val_loss += float(ce(vl, torch.from_numpy(y[take]).to(device))) * len(take)
        val_loss /= len(indices["val"])
        row = dict(epoch=epoch + 1, lr=rate, train_loss=total / seen, val_loss=val_loss)
        history.append(row)
        print(json.dumps(row), flush=True)
        if val_loss < best_loss - 1e-8:
            best_loss, stale, best = val_loss, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale >= patience:
            break
    model.load_state_dict(best)
    model.eval()
    model.apply(disable_observer)
    model.cpu()
    profile = model_profile(model, x.shape[2], x.shape[3])
    features = {}
    with torch.no_grad():
        for role, ix in indices.items():
            features[role] = np.concatenate(
                [
                    model.features(torch.from_numpy(x[ix[s : s + 128]])).numpy()
                    for s in range(0, len(ix), 128)
                ]
            )
    w, b = (
        model.head.weight_quant(model.head.linear.weight).detach().numpy(),
        model.head.linear.bias.detach().numpy(),
    )
    vl, vy = torch.tensor(features["val"] @ w.T + b), torch.tensor(y[indices["val"]])
    log_t = nn.Parameter(torch.zeros(()))
    temp_optimizer = torch.optim.Adam([log_t], lr=0.01)
    for _ in range(100):
        temp_optimizer.zero_grad()
        loss = nn.functional.cross_entropy(vl / log_t.exp(), vy)
        loss.backward()
        temp_optimizer.step()
        with torch.no_grad():
            log_t.clamp_(math.log(0.05), math.log(20.0))
    temperature = float(log_t.detach().exp())
    scale = float(model.feature_quant.calculate_qparams()[0].item())
    IntegerHead.from_float(w, b, scale, temperature).save(out / "integer_head.npz")
    mw, mb = (
        model.margin_head.weight_quant(model.margin_head.linear.weight).detach().numpy(),
        model.margin_head.linear.bias.detach().numpy(),
    )
    IntegerHead.from_float(mw, mb, scale, temperature).save(out / "margin_head.npz")
    bundle = dict(
        weights_float=w,
        bias_float=b,
        feature_scale=scale,
        temperature=temperature,
        classes=data["classes"],
        dropout_weights=model.dropout_head[1].weight.detach().numpy(),
        dropout_bias=model.dropout_head[1].bias.detach().numpy(),
    )
    for role, values in features.items():
        bundle[f"z_{role}"] = np.clip(np.rint(values / scale), -128, 127).astype(np.int8)
        for key in ("y", "subject", "placement", "session", "start_s"):
            if key in data:
                bundle[f"{key}_{role}"] = data[key][indices[role]]
    np.savez_compressed(out / "features.npz", **bundle)
    torch.save(
        dict(
            model=model.state_dict(),
            classes=classes,
            feature_dim=int(config.get("feature_dim", 128)),
            temperature=temperature,
            input_shape=list(x.shape[1:]),
        ),
        out / "model.pt",
    )
    metadata = dict(
        seed=seed,
        config=config,
        data_sha256=hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        torch=torch.__version__,
        numpy=np.__version__,
        profile=profile,
        temperature=temperature,
        feature_scale=scale,
        split=split,
        role_counts={k: len(v) for k, v in indices.items()},
        status="reference_reimplementation_not_original_checkpoint",
    )
    (out / "training.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    return metadata
