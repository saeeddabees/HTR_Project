import torch


def greedy_decode(log_probs, lengths, idx2char, blank=0):
    """Greedy CTC decoding.

    log_probs: tensor (T, N, C) - model output, T time steps, N batch, C classes
    lengths:   tensor (N,)      - number of valid time steps per sample
    Returns a list of N decoded strings.
    """
    best = log_probs.argmax(dim=2)  # (T, N): best class at each time step
    results = []
    for n in range(best.shape[1]):
        seq = best[: int(lengths[n]), n].tolist()
        out = []
        prev = None
        for s in seq:
            # Rule 1: merge repeated characters. Rule 2: drop blanks.
            if s != prev and s != blank:
                out.append(idx2char[s])
            prev = s
        results.append("".join(out))
    return results


if __name__ == "__main__":
    # Toy test: the model "says" h h _ e _ l l _ l o  (_ = blank = 0)
    idx2char = {1: "h", 2: "e", 3: "l", 4: "o"}
    steps = [1, 1, 0, 2, 0, 3, 3, 0, 3, 4]
    lp = torch.full((len(steps), 1, 5), -10.0)
    for t, s in enumerate(steps):
        lp[t, 0, s] = 0.0
    print(greedy_decode(lp, torch.tensor([len(steps)]), idx2char))  # expected: ['hello']