"""Train-independent experiment runner; no paper results are hard-coded."""

from pathlib import Path
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
import numpy as np
import pandas as pd
from .codec import Encoder, PeelDecoder, UncertaintySketch
from .conformal import HistogramCalibrator, calibration_curve, select_k_star
from .controller import SLAController, success_lower_bound
from .integer import IntegerHead, IntegerSoftmax, Q15, class_scores, top2_uncertainty
from .link import Link, Event, receive
from .metrics import summarize, selective_risk, auroc

METHODS = (
    "ours",
    "local_only",
    "local_conformal",
    "full_offload",
    "fixed_chunk",
    "ordered",
    "mc_dropout",
    "pooled",
    "no_debt",
    "float_head",
    "phone_only",
    "edge_assist",
    "balanced_degree",
    "direct_margin",
)
_PREFIX_CACHE = OrderedDict()


def stable_seed(seed, role, index, packet_id):
    return (
        int.from_bytes(
            hashlib.blake2s(f"{seed}:{role}:{index}:{packet_id}".encode(), digest_size=4).digest(),
            "little",
        )
        or 1
    )


@lru_cache(maxsize=2)
def load_bundle(path, mtime, size):
    with np.load(path, allow_pickle=False) as f:
        data = {k: f[k] for k in f.files}
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return data, digest.hexdigest()


class Evidence:
    def __init__(self, run_dir, config, method):
        self.run_dir, self.config, self.method = Path(run_dir), config, method
        source = self.run_dir / "features.npz"
        st = source.stat()
        self.data, self.digest = load_bundle(str(source.resolve()), st.st_mtime_ns, st.st_size)
        self.head = IntegerHead.load(self.run_dir / "integer_head.npz")
        self.margin_head = IntegerHead.load(self.run_dir / "margin_head.npz")
        self.softmax = IntegerSoftmax()
        self.classes, self.dimension = self.head.weights.shape
        enc = dict(config["encoder"])
        if method == "balanced_degree":
            enc.update(lambda1=0.1, lambda2=0.1)
        self.encoder = Encoder(dimension=self.dimension, **enc)
        self.order = np.argsort(-abs(self.data["weights_float"]).sum(0), kind="stable")
        self.kind = "direct_margin" if method == "direct_margin" else config["score"]
        self.sigmoid = np.rint(Q15 / (1 + np.exp(-np.arange(-4096, 4097) / 256))).astype(np.int32)

    def predict(self, z):
        if self.method == "float_head":
            float_logits = (
                z.astype(float) * float(self.data["feature_scale"]) @ self.data["weights_float"].T
                + self.data["bias_float"]
            ) / float(self.data["temperature"])
            e = np.exp(float_logits - float_logits.max(-1, keepdims=True))
            p, logits = (
                np.rint(e / e.sum(-1, keepdims=True) * Q15).astype(np.int32),
                np.rint(float_logits * 256).astype(np.int32),
            )
        else:
            logits = self.head.logits(z)
            p = self.softmax(logits)
        confidence = None
        if self.kind == "direct_margin":
            confidence = self.sigmoid[
                np.clip(self.margin_head.logits(z)[..., 0], -4096, 4096) + 4096
            ]
        return p, class_scores(p, self.kind, logits, confidence)

    def reconstruct(self, z, events, seed, role, index, sketch):
        if self.method in ("local_only", "local_conformal"):
            return z.copy(), self.dimension
        if self.method in ("full_offload", "mc_dropout"):
            return (z.copy(), self.dimension) if events else (np.zeros_like(z), 0)
        if self.method in ("ordered", "fixed_chunk"):
            result, known = np.zeros_like(z), np.zeros(len(z), bool)
            for event in events:
                width = (
                    int(self.config.get("ordered_coordinates", max(1, event.bytes - 24)))
                    if self.method == "ordered"
                    else max(1, event.bytes - 24)
                )
                if width < 1:
                    raise ValueError("ordered coordinate width must be positive")
                start = event.packet_id * width if self.method == "ordered" else 0
                ix = self.order[start : min(start + width, len(z))]
                result[ix], known[ix] = z[ix], True
            return result, int(known.sum())
        decoder = PeelDecoder(
            self.dimension, self.config.get("peel_tolerance", 2), self.config["max_k"]
        )
        for event in events:
            decoder.add(
                self.encoder.encode(
                    z,
                    stable_seed(seed, role, index, event.packet_id),
                    event.packet_id,
                    index,
                    sketch,
                )
            )
        return decoder.z, int(decoder.known.sum())

    def prefixes(self, role, seed):
        equivalence = {
            "no_debt": "ours",
            "pooled": "ours",
            "phone_only": "ours",
            "edge_assist": "ours",
            "local_only": "local_conformal",
            "mc_dropout": "full_offload",
        }
        key = (
            self.digest,
            role,
            seed,
            equivalence.get(self.method, self.method),
            json.dumps(self.config["encoder"], sort_keys=True),
            self.kind,
            self.config["max_k"],
            self.config.get("ordered_coordinates"),
            self.config.get("fixed_chunk_bytes"),
            (self.run_dir / "integer_head.npz").stat().st_mtime_ns,
        )
        if key in _PREFIX_CACHE:
            _PREFIX_CACHE.move_to_end(key)
            return _PREFIX_CACHE[key]
        zs = self.data[f"z_{role}"]
        scores = np.zeros((len(zs), self.config["max_k"] + 1, self.classes), np.int32)
        for i, z in enumerate(zs):
            for k in range(self.config["max_k"] + 1):
                size = (
                    self.config.get("fixed_chunk_bytes", self.config["encoder"]["packet_bytes"])
                    if self.method == "fixed_chunk"
                    else self.config["encoder"]["packet_bytes"]
                )
                events = [Event(j, float(j), float(j + 1), size) for j in range(k)]
                partial, _ = self.reconstruct(z, events, seed, role, i, bytes(16))
                scores[i, k] = self.predict(partial)[1]
        _PREFIX_CACHE[key] = scores
        while len(_PREFIX_CACHE) > 8:
            _PREFIX_CACHE.popitem(last=False)
        return scores


def evaluate(run_dir, out_dir, config, method="ours", seed=7, trace=None):
    if method not in METHODS:
        raise ValueError(f"unknown method: {method}")
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ev = Evidence(run_dir, config, method)
    kmax, alpha, bins, max_size = (
        config["max_k"],
        config["alpha"],
        config["histogram_bins"],
        config["max_set_size"],
    )
    val_scores = ev.prefixes("val", seed)
    if len(val_scores) < 4:
        raise ValueError("need at least four validation windows")
    cut = len(val_scores) // 2
    pilot = HistogramCalibrator(kmax, bins, alpha).fit(val_scores[:cut], ev.data["y_val"][:cut])
    curve = calibration_curve(pilot, val_scores[cut:], ev.data["y_val"][cut:])
    k_star = select_k_star(curve, 1 - alpha, max_size)
    if method in ("local_only", "local_conformal"):
        k_star = 0
    if method in ("full_offload", "fixed_chunk", "mc_dropout"):
        k_star = 1

    def mc_predict(z, role, i):
        rng = np.random.default_rng(stable_seed(seed, role, i, 65534))
        feature = z.astype(float) * float(ev.data["feature_scale"])
        probabilities = []
        for _ in range(config.get("mc_passes", 20)):
            logits = (
                feature * (rng.random(len(z)) >= 0.2) / 0.8 @ ev.data["dropout_weights"].T
                + ev.data["dropout_bias"]
            )
            p = np.exp(logits - logits.max())
            probabilities.append(p / p.sum())
        return np.mean(probabilities, 0), float(np.var(probabilities, axis=0).mean())

    mc_threshold = 1.0
    if method == "mc_dropout":
        vp = np.stack([mc_predict(z, "val", i)[0] for i, z in enumerate(ev.data["z_val"])])
        choices = [
            v
            for v in np.unique(vp.max(-1))
            if (vp.argmax(-1)[vp.max(-1) >= v] == ev.data["y_val"][vp.max(-1) >= v]).mean()
            >= 1 - alpha
        ]
        mc_threshold = float(min(choices)) if choices else 1.0

    def run_role(role):
        cfg = config
        kwargs = dict(cfg["controller"])
        kwargs.update(
            max_k=kmax,
            debt_enabled=method != "no_debt",
            base_energy_uj=cfg["energy"]["backbone_uj"]
            + cfg["deadline_ms"] * cfg["energy"]["idle_uj_per_ms"],
        )
        controller = SLAController(**kwargs)
        link = Link(cfg["regime"], stable_seed(seed, role, 0, 65535), cfg["jitter_ms"], trace)
        sketch = UncertaintySketch()
        zs, labels = ev.data[f"z_{role}"], ev.data[f"y_{role}"]
        scores, rows = [], []
        for i, z in enumerate(zs):
            wire_sketch = sketch.update(int(top2_uncertainty(ev.predict(z)[0])))
            compute, decode = cfg["timing"]["backbone_ms"], cfg["timing"]["decode_ms"]
            if method == "edge_assist":
                decode = cfg["timing"]["edge_decode_ms"] + cfg["timing"]["edge_forward_ms"]
            if method == "phone_only":
                decode = cfg["timing"]["phone_busy_decode_ms"]
            size = cfg["encoder"]["packet_bytes"]
            if method == "ordered":
                size = int(cfg.get("ordered_packet_bytes", size))
            if method == "fixed_chunk":
                size = int(cfg.get("fixed_chunk_bytes", size))
            if method in ("full_offload", "mc_dropout"):
                size = ev.dimension + 24
            controller.packet_energy_uj = (
                cfg["energy"]["encode_uj"] + size * cfg["energy"]["tx_uj_per_byte"]
            )
            decision = controller.decide(
                k_star, link.q, link.rate * 64 / size, cfg["deadline_ms"] - compute - decode
            )
            count, target = decision.emissions, decision.target_k
            if method in ("local_only", "local_conformal"):
                count, target = 0, 0
            elif method in ("full_offload", "fixed_chunk", "mc_dropout"):
                count, target = 1, 1
            elif method == "ordered":
                width = int(
                    cfg.get("ordered_coordinates", max(1, cfg["encoder"]["packet_bytes"] - 24))
                )
                count = min(count, int(np.ceil(ev.dimension / width)))
                target = min(target, max(1, count))
            potential = link.generate(i, count, compute, size)
            if target == 0:
                events, emitted, latency = [], [], compute
            else:
                events, emitted, latency = receive(
                    potential,
                    cfg["deadline_ms"],
                    target,
                    decode,
                    cfg["timing"]["ack_delay_ms"],
                    kmax,
                )
            partial, resolved = ev.reconstruct(z, events, seed, role, i, wire_sketch)
            p, score = ev.predict(partial)
            k = len(events)
            debt = controller.debt
            controller.observe(int(pilot.predict(score, k).sum()), max_size)
            byte_count = sum(e.bytes for e in emitted)
            radio = byte_count * cfg["energy"]["tx_uj_per_byte"]
            encoding = (
                len(emitted) * cfg["energy"]["encode_uj"]
                if method
                not in (
                    "local_only",
                    "local_conformal",
                    "full_offload",
                    "fixed_chunk",
                    "ordered",
                    "mc_dropout",
                )
                else 0.0
            )
            idle = latency * cfg["energy"]["idle_uj_per_ms"]
            energy = cfg["energy"]["backbone_uj"] + encoding + radio + idle
            winner, confidence, variance = int(p.argmax()), None, None
            if method == "mc_dropout":
                mp, variance = mc_predict(partial, role, i)
                winner, confidence = int(mp.argmax()), float(mp.max())
            budget_n = min(
                count,
                max(0, int((cfg["deadline_ms"] - compute - decode) * link.rate * 64 / size / 1000)),
            )
            row = dict(
                window_id=i,
                subject=str(ev.data[f"subject_{role}"][i]),
                placement=str(ev.data.get(f"placement_{role}", np.repeat("unknown", len(zs)))[i]),
                label=int(labels[i]),
                winner=winner,
                k=k,
                resolved=resolved,
                target_k=target,
                emitted=len(emitted),
                bytes_sent=byte_count,
                latency_ms=latency,
                deadline_ms=cfg["deadline_ms"],
                energy_uj=energy,
                compute_energy_uj=cfg["energy"]["backbone_uj"] + encoding,
                radio_energy_uj=radio,
                idle_energy_uj=idle,
                energy_cap_uj=controller.energy_cap_uj,
                airtime_ms=byte_count * 8 / (cfg["phy_mbps"] * 1000)
                + len(emitted) * cfg["packet_overhead_ms"],
                correct=int(winner == int(labels[i])),
                uncertainty=float(top2_uncertainty(p)) / Q15,
                debt=debt,
                bound=success_lower_bound(budget_n, link.q, target),
                budget_feasible=int(decision.feasible),
                budget_reason=decision.reason,
                evidence_success=int(k >= target and k_star is not None),
                mc_confidence=confidence,
                mc_variance=variance,
                source="trace_replay" if trace else "simulation",
            )
            scores.append(score)
            rows.append(row)
        return np.stack(scores), rows

    if config.get("record_policy_tuning_stream", False):
        _, tuning = run_role("val")
        pd.DataFrame(tuning).to_csv(out / "tuning_windows.csv", index=False)
    cal_scores, cal_rows = run_role("cal")
    cal_k = np.array([r["k"] for r in cal_rows], int)
    tensor = np.zeros((len(cal_k), kmax + 1, ev.classes), np.int32)
    tensor[np.arange(len(cal_k)), cal_k] = cal_scores
    cal = HistogramCalibrator(kmax, bins, alpha, method == "pooled").fit(
        tensor, ev.data["y_cal"], cal_k
    )
    cal.save(out / "calibration.npz")
    test_scores, rows = run_role("test")
    for score, row in zip(test_scores, rows):
        if method in ("local_only", "full_offload", "mc_dropout"):
            prediction = np.zeros(ev.classes, bool)
            accepted = method == "local_only" or row["k"] > 0
            if method == "mc_dropout":
                accepted = accepted and row["mc_confidence"] >= mc_threshold
            if accepted:
                prediction[row["winner"]] = True
        else:
            prediction = cal.predict(score, row["k"])
        row.update(
            prediction_set=json.dumps(np.flatnonzero(prediction).tolist()),
            set_size=int(prediction.sum()),
            covered=int(prediction[row["label"]]),
        )
    frame = pd.DataFrame(rows)
    frame.to_csv(out / "windows.csv", index=False)
    pd.DataFrame(curve).to_csv(out / "policy_tuning_curve.csv", index=False)
    pd.DataFrame(selective_risk(rows)).to_csv(out / "selective_risk.csv", index=False)
    pk = [
        dict(
            k=int(k),
            n=len(g),
            coverage=float(g.covered.mean()),
            set_size=float(g.set_size.mean()),
            threshold_q15=cal.threshold(int(k)),
        )
        for k, g in frame.groupby("k")
    ]
    pd.DataFrame(pk).to_csv(out / "per_k.csv", index=False)
    frame.groupby("label", as_index=False).agg(
        n=("covered", "size"), coverage=("covered", "mean"), mean_set_size=("set_size", "mean")
    ).to_csv(out / "per_class.csv", index=False)
    metrics = summarize(rows, config["stride_s"])
    if config.get("id_placement"):
        metrics["placement_ood_auroc"] = auroc(
            frame.placement != config["id_placement"], frame.uncertainty
        )
    metadata = dict(
        method=method,
        seed=seed,
        config=config,
        metrics=metrics,
        k_star=k_star,
        mc_threshold=mc_threshold if method == "mc_dropout" else None,
        source="trace_replay" if trace else "simulation",
        energy_source="configured_cost_model_not_hardware_measurement",
        calibration_group_counts=np.bincount(cal_k, minlength=kmax + 1).tolist(),
        validity="empirical under subject shift/adaptive debt; see docs/REPRODUCIBILITY.md",
        features_sha256=ev.digest,
    )
    (out / "summary.json").write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return metadata
