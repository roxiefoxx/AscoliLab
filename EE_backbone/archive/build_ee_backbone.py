import csv

INPUT = "mij_matrix.csv"
OUTPUT_EDGES = "ee_backbone_greedy_edges.csv"

EXCITATORY_RULE = lambda lab: (
    ("Mossy Fiber Associated" not in lab)
    and ("Interneuron" not in lab)
    and (
        "Pyramidal" in lab
        or "Granule" in lab
        or "Stellate" in lab
        or "Principal" in lab
        or "Fan" in lab
        or lab == "DG Mossy"
        or "Back Projection" in lab
    )
)

with open(INPUT, newline="") as f:
    rows = list(csv.reader(f))

labels = rows[0][1:]
senders = [r[0] for r in rows[1:]]
assert labels == senders, "Row/column labels differ; align matrix before running."

M = [[float(x) if x.strip() else 0.0 for x in r[1:]] for r in rows[1:]]
E = [lab for lab in labels if EXCITATORY_RULE(lab)]
idx = {lab: i for i, lab in enumerate(labels)}
unvisited = {idx[lab] for lab in E}

edges = []
components = []
while unvisited:
    # Start next path at strongest positive E->E edge among remaining nodes.
    best = None
    for i in unvisited:
        for j in unvisited:
            if i != j and M[i][j] > 0:
                if best is None or M[i][j] > best[2]:
                    best = (i, j, M[i][j])

    if best is None:
        # Isolated remaining E node.
        i = min(unvisited)
        components.append([labels[i]])
        unvisited.remove(i)
        continue

    cid = len(components) + 1
    i, j, w = best
    path = [labels[i], labels[j]]
    unvisited.remove(i)
    unvisited.remove(j)
    edges.append([cid, 1, labels[i], labels[j], w])

    current = j
    step = 2
    while True:
        candidates = [(M[current][k], k) for k in unvisited if M[current][k] > 0]
        if not candidates:
            break
        w, k = max(candidates)
        edges.append([cid, step, labels[current], labels[k], w])
        path.append(labels[k])
        unvisited.remove(k)
        current = k
        step += 1

    components.append(path)

with open(OUTPUT_EDGES, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["component", "step", "sender", "receiver", "weight"])
    writer.writerows(edges)

print(f"E nodes: {len(E)}")
print(f"Backbone edges: {len(edges)}")
print(f"Components: {len(components)}")
