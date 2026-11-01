import argparse
import glob
import json
import os
import re
from collections import Counter, defaultdict

# Features that are too generic to say anything about a topic.
GENERIC = {"@the", "@youtube", "@addthis", "#ff", "#fb", "#followfriday", "#rt"}
VALID = re.compile(r"^[#@][a-z][a-z0-9_]{2,}$")  # drops "#1", "#...", "@..." etc.

def user_features(folder):
    feats = defaultdict(set)
    def add(node, names, values):
        for name, val in zip(names, values):
            if val == "1":
                name = name.lower().rstrip(".,:;!?")
                if VALID.match(name):
                    feats[node].add(name)

    for path in glob.glob(os.path.join(folder, "*.featnames")):
        ego = os.path.basename(path)[: -len(".featnames")]
        base = os.path.join(folder, ego)
        names = [l.split(" ", 1)[1].strip() for l in open(path, encoding="utf-8") if l.strip()]
        if os.path.exists(base + ".feat"):
            for line in open(base + ".feat"):
                parts = line.split()
                if parts:
                    add(parts[0], names, parts[1:])
        if os.path.exists(base + ".egofeat"):
            add(ego, names, open(base + ".egofeat").read().split())
    return feats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/twitter")
    ap.add_argument("--comm", default="commMetrics.json")
    ap.add_argument("--top", type=int, default=15, help="number of largest communities")
    ap.add_argument("--min-share", type=float, default=0.04, help="feature must be used by at least this share of the community")
    ap.add_argument("--out", default="community_topics.json")
    args = ap.parse_args()
    feats = user_features(args.data)
    membership = json.load(open(args.comm))["leiden"]["membership"]
    sizes = Counter(membership.values())
    users = [v for v in membership if feats.get(v)]
    overall = Counter(f for v in users for f in feats[v])
    n_all = len(users)
    by_comm = defaultdict(list)
    for v in users:
        by_comm[membership[v]].append(v)
    result = {}
    for cid, size in sizes.most_common(args.top):
        members = by_comm[cid]
        n = len(members)
        counts = Counter(f for v in members for f in feats[v])
        rows = []
        for f, k in counts.items():
            if f in GENERIC or k < max(10, args.min_share * n):
                continue
            share = k / n
            lift = share / (overall[f] / n_all)
            rows.append({"feature": f, "share": share, "lift": lift})
        rows.sort(key=lambda r: r["lift"] * r["share"] ** 0.5, reverse=True)
        result[cid] = {"size": size, "top": rows[:8]}
        tops = ", ".join(f"{r['feature']} ({100 * r['share']:.0f}%, x{r['lift']:.1f})" for r in rows[:8])
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)


if __name__ == "__main__":
    main()
