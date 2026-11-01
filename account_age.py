import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"

def main():
    m = json.load(open("metrics.json"))
    n = m["numNodes"]
    users = list(m["inDegreeCentrality"])
    ids = np.array([int(u) for u in users])
    followers = np.array([round(m["inDegreeCentrality"][u] * (n - 1)) for u in users])
    following = np.array([round(m["outDegreeCentrality"][u] * (n - 1)) for u in users])
    keep = (followers + following) > 0
    ids, followers, following = ids[keep], followers[keep], following[keep]
    order = np.argsort(ids)
    ids, followers, following = ids[order], followers[order], following[order]
    groups = np.array_split(np.arange(len(ids)), 10)
    rho_in, p_in = spearmanr(ids, followers)
    rho_out, p_out = spearmanr(ids, following)
    popular = followers >= 1000
    oldest20 = np.zeros(len(ids), bool)
    oldest20[np.concatenate(groups[:2])] = True
    result = {
        "numUsers": int(len(ids)),
        "spearmanIdFollowers": [float(rho_in), float(p_in)],
        "spearmanIdFollowing": [float(rho_out), float(p_out)],
        "usersWith1000Followers": int(popular.sum()),
        "shareOfThoseInOldest20pct": float(oldest20[popular].mean()),
        "groups": [],
    }
    for i, g in enumerate(groups, 1):
        row = {"group": i, "idFrom": int(ids[g[0]]), "idTo": int(ids[g[-1]]),
               "medianFollowers": float(np.median(followers[g])), "meanFollowers": float(followers[g].mean()), "medianFollowing": float(np.median(following[g])), "meanFollowing": float(following[g].mean())}
        result["groups"].append(row)
    json.dump(result, open("account_age.json", "w"), indent=2)
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    x = np.arange(1, 11)
    for key, label, colour in [("meanFollowers", "Followers (in-degree)", BLUE), ("meanFollowing", "Following (out-degree)", ORANGE)]:
        y = [r[key] for r in result["groups"]]
        ax.plot(x, y, color=colour, lw=2, marker="o", ms=5, markeredgecolor="#fcfcfb", markeredgewidth=1.5, label=label)
        ax.annotate(label, (10, y[-1]), xytext=(8, 0), textcoords="offset points", va="center", fontsize=8, color=INK)
    ax.set_xticks(x)
    ax.set_xticklabels(["1\noldest"] + [str(i) for i in range(2, 10)] + ["10\nnewest"])
    ax.set_xlabel("Account-age group (10% of users each, by user ID)", color=MUTED)
    ax.set_ylabel("Mean number of accounts", color=MUTED)
    ax.set_ylim(0, None)
    ax.set_xlim(0.6, 12.6)
    ax.grid(axis="y", color=GRID, lw=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    ax.spines["left"].set_color(GRID)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    ax.legend(frameon=False, loc="upper center", ncol=2, fontsize=8, labelcolor=MUTED)
    fig.tight_layout()
    fig.savefig("report_figures/22_account_age.pdf", bbox_inches="tight")
    fig.savefig("report_figures/22_account_age.png", dpi=220, bbox_inches="tight")


if __name__ == "__main__":
    main()
