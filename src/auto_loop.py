"""Level A of the self-improving loop (see docs/auto-config.md): automate the
propose -> test -> keep ablation we ran by hand over the toolbox.

Hill-climb the tool set. From a champion config, try single-axis neighbors (one
tool flipped) on a short training budget, evaluate each on the game's own metric,
and adopt any neighbor that beats the champion — then repeat. Every candidate's
config and score is logged, so the run *is* the meta-analysis (layer 3), automatic.

    python src/auto_loop.py --game arc_ls20 --budget 150000 --rounds 2

Compute-aware: short budget per candidate (triage), sequential (CPU-bound). The
winner should later get a full-length run. Runs long — launch in the background.
"""
import perf  # first: BLAS thread limits

import argparse
import json
import os
import re
import subprocess

APP = "/app"
CKPT_GLOB = "data/models/{game}_ppo_spr*_[0-9]*_steps.zip"

# the axes the loop searches; the champion is the best config known so far
CHAMPION = {"sil": 1.0, "intrinsic": 1.0, "spr": 0.0, "fs": 1, "auto_gamma": True}
NEIGHBORS = {                      # one flip each -> candidate configs
    "sil": [0.0],                  # is self-imitation actually pulling weight?
    "spr": [1.0],                  # does a dynamics-aware encoder help here?
    "intrinsic": [0.0],            # is curiosity needed?
    "fs": [4],                     # motion perception (usually not, on grids)
    "auto_gamma": [False],         # does the horizon-tuned discount matter?
}


def flags(cfg):
    out = ["--spr-coef", str(cfg["spr"]), "--sil-coef", str(cfg["sil"]),
           "--intrinsic-coef", str(cfg["intrinsic"]), "--frame-stack", str(cfg["fs"])]
    if cfg["auto_gamma"]:
        out.append("--auto-gamma")
    return out


def evaluate(game, cfg, budget, n_envs, seed, episodes, tag):
    out = f"/app/data/models/_auto_{tag}"
    train = (["python", "src/train_ppo_spr.py", "--game", game, "--timesteps",
              str(budget), "--n-envs", str(n_envs), "--seed", str(seed), "--out", out]
             + flags(cfg))
    with open(f"/app/data/logs/auto_{tag}.log", "w") as log:
        subprocess.run(train, cwd=APP, stdout=log, stderr=subprocess.STDOUT, check=False)
    score = -1.0                                    # -1 = train/eval failed
    for _ in range(3):                              # retry: the ARC SDK's Arcade()
        ev = subprocess.run(                        # network setup can fail transiently
            ["python", "src/eval_arc.py", out + ".zip", game, str(episodes), str(cfg["fs"])],
            cwd=APP, capture_output=True, text=True)   # frame-stack must match training
        m = re.search(r"mean ([0-9.]+)", ev.stdout)
        if m:
            score = float(m.group(1))
            break
    subprocess.run(f"rm -f {out}* " + CKPT_GLOB.format(game=game),
                   cwd=APP, shell=True, check=False)
    return score


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--game", default="arc_ls20")
    ap.add_argument("--budget", type=int, default=150_000, help="steps per candidate")
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n-envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=20)
    args = ap.parse_args()
    perf.setup_cpu_threads(None)
    os.makedirs("/app/data/logs", exist_ok=True)

    results_path = f"/app/data/logs/auto_loop_{args.game}.jsonl"
    results = open(results_path, "w")

    def record(cfg, score, note):
        results.write(json.dumps({"cfg": cfg, "score": score, "note": note}) + "\n")
        results.flush()
        print(f"[auto-loop] {note}: score={score:.3f}  cfg={cfg}", flush=True)

    champion = dict(CHAMPION)
    best = evaluate(args.game, champion, args.budget, args.n_envs, args.seed,
                    args.episodes, "champion0")
    record(champion, best, "champion (start)")

    for r in range(args.rounds):
        improved = False
        for axis, values in NEIGHBORS.items():
            for v in values:
                if champion[axis] == v:
                    continue
                cand = dict(champion)
                cand[axis] = v
                tag = f"r{r}_{axis}_{v}"
                score = evaluate(args.game, cand, args.budget, args.n_envs, args.seed,
                                 args.episodes, tag)
                record(cand, score, f"round {r} try {axis}={v}")
                if score > best:
                    best, champion, improved = score, cand, True
                    record(champion, best, f"round {r} NEW champion")
        if not improved:
            print(f"[auto-loop] round {r}: no improvement, stopping.", flush=True)
            break

    print(f"\n[auto-loop] DONE. best score={best:.3f}\n  champion={champion}", flush=True)
    results.close()


if __name__ == "__main__":
    main()
