import csv

INPUT = "mij_matrix.csv"

def is_excitatory(label):
    # Heuristic used for this matrix's cell-class labels.
    return (
        ("Mossy Fiber Associated" not in label)
        and ("Interneuron" not in label)
        and (
            "Pyramidal" in label
            or "Granule" in label
            or "Stellate" in label
            or "Principal" in label
            or "Fan" in label
            or label == "DG Mossy"
            or "Back Projection" in label
        )
    )

with open(INPUT, newline="") as f:
    rows = list(csv.reader(f))

labels = rows[0][1:]
senders = [r[0] for r in rows[1:]]
if labels != senders:
    raise ValueError("Row and column labels differ; align the matrix before running.")

M = [[float(x) if x.strip() else 0.0 for x in row[1:]] for row in rows[1:]]
idx = {label: i for i, label in enumerate(labels)}
E = [label for label in labels if is_excitatory(label)]
E_idx = [idx[label] for label in E]

all_edges = []
summary = []
paths = {}

for seed in E_idx:
    visited = {seed}
    path = [seed]
    current = seed
    step = 1
    total_weight = 0.0

    while True:
        candidates = [
            (M[current][k], k)
            for k in E_idx
            if k not in visited and M[current][k] > 0
        ]
        if not candidates:
            break

        # Highest outgoing E->E edge; deterministic tie-break by receiver label.
        candidates.sort(key=lambda x: (-x[0], labels[x[1]]))
        weight, nxt = candidates[0]

        all_edges.append([labels[seed], step, labels[current], labels[nxt], weight])
        total_weight += weight
        visited.add(nxt)
        path.append(nxt)
        current = nxt
        step += 1

    paths[labels[seed]] = [labels[i] for i in path]
    summary.append([
        labels[seed],
        len(path),
        len(path) - 1,
        total_weight,
        labels[path[-1]],
        len(E) - len(path),
    ])

with open("ee_backbone_per_injection_edges.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["injection_seed", "step", "sender", "receiver", "weight"])
    writer.writerows(all_edges)

with open("ee_backbone_per_injection_summary.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow([
        "injection_seed",
        "nodes_reached_including_seed",
        "edges_in_path",
        "total_path_weight",
        "terminal_node",
        "unreached_E_nodes",
    ])
    writer.writerows(summary)

with open("ee_backbone_per_injection_paths.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["injection_seed", "ordered_path"])
    for seed, path in paths.items():
        writer.writerow([seed, " -> ".join(path)])

print(f"Excitatory seeds: {len(E)}")
print(f"Total seed-specific backbone edges: {len(all_edges)}")
