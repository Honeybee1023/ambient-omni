"""Nested space-filling design over the manual schedule box (2026-09-14).

A schedule in the user's parameterisation is four numbers:
    b     where the T=0 stretch ends          in [0, 0.9]
    r     bend position, as a fraction of the way from b to the end, in [0.1, 0.9]
          (bend at x_m = b + r (1 - b))
    y     bend height T(x_m)                  in [0, 1]
    T_end final T                             in [0.8, 1.0]
giving control points [[0,0],[b,0],[x_m,y],[1,T_end]].

The design is greedy farthest-point sampling in the box normalised to [0,1]^4,
starting from every schedule already trained (or queued) that lies exactly in
this family. Each pick is the candidate farthest from everything so far, so any
prefix of the list is a uniform cover and the next pick always fills the
biggest remaining gap -- stage 1 = picks 1-16, stage 2 = 17-32, stage 3 = 33-64.
Candidates are the 16 box corners plus a scrambled Sobol pool (rounded to 0.01).

Two kinds of box point describe the same schedule, and distances respect that:
  * a straight rise (y = r * T_end): every r gives the same curve, so it covers
    a whole line in the box;
  * a flat bend (y = 0): the curve is "T=0 until x_m, then straight", which any
    (b', r') with the same x_m also draws, as does the straight rise from x_m.
More generally, a candidate whose T(p) curve is within TAU (RMS over training)
of a trained, queued or picked schedule is a near-duplicate: its box location
counts as covered and it is never queued. TAU = 0.06 is half the curve distance
between hold50 and the b50 concave centre (0.11), two schedules that measured
the same MIND at n=2. Reference curves outside the family (static T=0, the two
step runs) take part only in this check.
Queued-but-unstarted runs compete as candidates and are preferred when they are
at least PREFER of the best distance (they are already in the manifest).

Usage (needs numpy + scipy; run on a cluster):
    python grid_design.py --out grid_design.json
"""
import argparse, json
import numpy as np
from scipy.stats import qmc

LO = np.array([0.0, 0.1, 0.0, 0.8]); HI = np.array([0.9, 0.9, 1.0, 1.0])
PREFER = 0.8
TAU = 0.06
PGRID = np.linspace(0, 1, 4001)
REF_CURVES = {  # trained, outside the family; used only for near-duplicate checks
    "v2_static_T000": [[0, 0], [1, 0]],
    "shape_b50_step": [[0, 0], [0.5, 0], [0.5, 0.95], [1, 0.95]],
    "shape_b25_step": [[0, 0], [0.25, 0], [0.25, 0.95], [1, 0.95]],
}
STAGES = [(1, 16), (2, 16), (3, 32)]

# Trained runs that are exactly in the family. Straight rises are given at r=0.5.
DONE = [  # name, b, r, y, T_end
    ("shape_b50_concave",      0.50, 0.5, 0.75,  0.95),
    ("shape_b50_convex",       0.50, 0.5, 0.20,  0.95),
    ("sched_hold50_ceil95",    0.50, 0.5, 0.475, 0.95),
    ("p1_s_late_hard",         0.50, 0.5, 0.40,  0.95),
    ("shape_b25_concave",      0.25, 0.5, 0.75,  0.95),
    ("shape_b25_convex",       0.25, 0.5, 0.20,  0.95),
    ("p0_warmup_pw5",          0.25, 0.5, 0.475, 0.95),
    ("v2_warmup15_0to095",     0.15, 0.5, 0.475, 0.95),
    ("v2_warmup40_0to095",     0.40, 0.5, 0.475, 0.95),
    ("v2_warmup25_0to085",     0.25, 0.5, 0.425, 0.85),
    ("v2_linear_0to095",       0.00, 0.5, 0.475, 0.95),
    ("v2_linear_0to085",       0.00, 0.5, 0.425, 0.85),
    ("p1_s_ceiling_100",       0.00, 0.5, 0.50,  1.00),
    ("v2_twophase_0_050_095",  0.00, 0.5, 0.50,  0.95),
]
# Queued (not started) runs in the family.
QUEUED = [
    ("man_b50_bend85",         0.50, 0.5, 0.85,  0.95),
    ("man_b50_bend95",         0.50, 0.5, 0.95,  0.95),
    ("man_b50_bendearly",      0.50, 0.25, 0.75, 0.95),
    ("man_b50_concave_end100", 0.50, 0.5, 0.75,  1.00),
    ("man_b40_concave",        0.40, 0.5, 0.75,  0.95),
    ("man_b60_concave",        0.60, 0.5, 0.75,  0.95),
    ("man_b75_linear",         0.75, 0.5, 0.475, 0.95),
    ("man_b90_linear",         0.90, 0.5, 0.475, 0.95),
    ("man_b75_concave",        0.75, 0.5, 0.75,  0.95),
    ("man_b50_end80",          0.50, 0.5, 0.40,  0.80),
    ("man_b50_end90",          0.50, 0.5, 0.45,  0.90),
    ("man_b25_end100",         0.25, 0.5, 0.50,  1.00),
    ("man_b50_overshoot",      0.50, 0.4, 1.00,  0.90),
]
OFF_BOX = ["man_noblade_030", "man_b25_dip", "man_b10_plateau50"]  # not in the family


def norm(p):
    return (np.asarray(p, float) - LO) / (HI - LO)


def equivalents(p, k=33):
    """All box points that draw the same schedule as p (including p)."""
    b, r, y, T = p
    out = [p]
    rs = np.linspace(0.1, 0.9, k)
    if abs(y - r * T) < 1e-6:                       # straight rise from b
        out += [(b, rr, rr * T, T) for rr in rs]
    if y == 0:                                      # flat until x_m, then straight
        xm = b + r * (1 - b)
        for bb in np.linspace(0, min(xm, 0.9), k):
            rr = (xm - bb) / (1 - bb)
            if 0.1 <= rr <= 0.9:
                out.append((bb, rr, 0.0, T))
        if xm <= 0.9:
            out += [(xm, rr, rr * T, T) for rr in rs]
    return np.array([norm(q) for q in out])


def points(b, r, y, T):
    xm = round(b + r * (1 - b), 4)
    cp = [[0.0, 0.0]] + ([[b, 0.0]] if b > 0 else []) + [[xm, y], [1.0, T]]
    return cp


def curve(cp):
    xs = [c[0] for c in cp]; ys = [c[1] for c in cp]
    return np.interp(PGRID, xs, ys)


def bins_table(rows, i, j, labels, nb=4):
    """Markdown table of how many points fall in each (param i, param j) cell, by group.
    Group "dup" = Sobol-pool schedules in that cell that are near-duplicates of a trained or picked one."""
    keys = ["have", "s1", "s2", "s3", "dup"]
    cnt = {}
    for r in rows:
        u = norm(r["p"])
        ci, cj = min(int(u[i] * nb), nb - 1), min(int(u[j] * nb), nb - 1)
        cnt.setdefault((ci, cj), {k: 0 for k in keys})[r["g"]] += 1
    edges = lambda k: [f"{LO[k] + (HI[k]-LO[k])*q/nb:.2f}-{LO[k] + (HI[k]-LO[k])*(q+1)/nb:.2f}" for q in range(nb)]
    out = [f"| {labels[i]} \\ {labels[j]} | " + " | ".join(edges(j)) + " |", "|---" * (nb + 1) + "|"]
    for ci, e in enumerate(edges(i)):
        cells = []
        for cj in range(nb):
            c = cnt.get((ci, cj))
            cells.append("." if not c else "/".join(str(c[k]) for k in keys))
        out.append(f"| {e} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def name_for(i, b, r, y, T):
    return f"man_g{i:02d}_b{round(b*100):02d}_r{round(r*100):02d}_y{round(y*100):03d}_T{round(T*100):03d}"


def radius(sample_sets, probes):
    S = np.concatenate(sample_sets)
    d = np.full(len(probes), np.inf)
    for i in range(0, len(S), 512):
        d = np.minimum(d, np.linalg.norm(probes[:, None, :] - S[None, i:i + 512, :], axis=2).min(1))
    j = int(d.argmax())
    return float(d[j]), (LO + probes[j] * (HI - LO)).round(2).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="grid_design.json")
    ap.add_argument("--seed", type=int, default=20260914)
    a = ap.parse_args()

    corners = [tuple(LO + np.array(c) * (HI - LO)) for c in np.ndindex(2, 2, 2, 2)]
    sob = LO + qmc.Sobol(4, scramble=True, seed=a.seed).random(4096) * (HI - LO)
    sob = np.round(sob, 2)
    cands = [("corner", tuple(map(float, c))) for c in corners] + [("sobol", tuple(map(float, s))) for s in sob]
    cands += [("queued:" + n, (b, r, y, T)) for n, b, r, y, T in QUEUED]
    probes = qmc.Sobol(4, scramble=True, seed=a.seed + 1).random(16384)

    present = [equivalents(p[1:]) for p in DONE]
    samp = [equivalents(c[1]) for c in cands]
    owner = np.concatenate([np.full(len(s), i) for i, s in enumerate(samp)])
    flat = np.concatenate(samp)
    mind = np.full(len(flat), np.inf)

    def update(S):
        nonlocal mind
        mind = np.minimum(mind, np.linalg.norm(flat[:, None, :] - S[None, :, :], axis=2).min(1))

    cc = np.array([curve(points(*c[1])) for c in cands])
    taken = set()
    dup_of = {}

    def cover_duplicates(cv, label):
        rms = np.sqrt(((cc - cv) ** 2).mean(1))
        for i in np.where(rms < TAU)[0]:
            if int(i) in taken:
                continue
            taken.add(int(i)); dup_of[int(i)] = label
            update(samp[i]); present.append(samp[i])

    for S in list(present):
        update(S)
    for d in DONE:
        cover_duplicates(curve(points(*d[1:])), d[0])
    for n, cp in REF_CURVES.items():
        cover_duplicates(curve(cp), n)
    report = {"radius_done_only": radius(present, probes)}
    picks = []
    total = sum(n for _, n in STAGES)
    stage_of = [s for s, n in STAGES for _ in range(n)]
    for k in range(total):
        score = np.full(len(cands), np.inf)
        np.minimum.at(score, owner, mind)
        score[list(taken)] = -1
        best = int(score.argmax())
        q = [i for i, c in enumerate(cands) if c[0].startswith("queued:") and i not in taken]
        if q:
            qi = max(q, key=lambda i: score[i])
            if score[qi] >= PREFER * score[best]:
                best = qi
        taken.add(best)
        src, (b, r, y, T) = cands[best]
        nm = src.split(":", 1)[1] if src.startswith("queued:") else name_for(k + 1, b, r, y, T)
        picks.append({"order": k + 1, "stage": stage_of[k], "name": nm, "source": src.split(":")[0],
                      "b": b, "r": r, "y": y, "T_end": T,
                      "control_points": points(b, r, y, T), "gap_filled": round(float(score[best]), 3)})
        update(samp[best]); present.append(samp[best])
        cover_duplicates(cc[best], nm)
        if k + 1 in np.cumsum([n for _, n in STAGES]):
            report[f"radius_after_stage{stage_of[k]}"] = radius(present, probes)
    picked = {p["name"] for p in picks}
    dupnames = {cands[i][0].split(":", 1)[1] for i in dup_of if cands[i][0].startswith("queued:")}
    rest = [n for n, *_ in QUEUED if n not in picked]
    tail = [n for n in rest if n not in dupnames] + [n for n in rest if n in dupnames] + OFF_BOX
    report["corners_covered_as_duplicates"] = {str(cands[i][1]): dup_of[i] for i in range(16) if i in dup_of}
    report["queued_covered_as_duplicates"] = {cands[i][0]: dup_of[i] for i in dup_of if cands[i][0].startswith("queued:")}
    rows = [{"p": cands[i][1], "g": "dup"} for i in dup_of if cands[i][0] == "sobol"]
    rows += [{"p": d[1:], "g": "have"} for d in DONE] + [{"p": (p["b"], p["r"], p["y"], p["T_end"]), "g": f"s{p['stage']}"} for p in picks]
    labels = ["b", "r", "y", "T_end"]
    tables = [f"#### {labels[i]} x {labels[j]} (cell = have/stage1/stage2/stage3/pool-duplicates)\n" + bins_table(rows, i, j, labels)
              for i in range(4) for j in range(i + 1, 4)]
    out = {"box": {"b": [0, 0.9], "r": [0.1, 0.9], "y": [0, 1], "T_end": [0.8, 1.0]},
           "seed": a.seed, "prefer_queued": PREFER, "done": [dict(zip(["name", "b", "r", "y", "T_end"], d)) for d in DONE],
           "picks": picks, "tail": tail, "coverage": report, "tau": TAU,
           "projection_tables_md": "\n\n".join(tables)}
    json.dump(out, open(a.out, "w"), indent=1)
    for p in picks:
        print(p["order"], p["stage"], p["name"], p["control_points"], p["gap_filled"])
    print("tail:", tail)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
