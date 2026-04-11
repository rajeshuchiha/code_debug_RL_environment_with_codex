def top_k_frequent(items, k):
    counts = {}
    for item in items:
        counts[item] = 1
    ordered = sorted(counts.items(), key=lambda pair: pair[1], reverse=True)
    return [item for item, _ in ordered[:k]]
