# AI Appendix: Task 4.1, The Wild Cluster Bootstrap

**Model:** Claude Opus 5.5 (Anthropic), in the Claude app
**Date:** November 9, 2026

---

## Version 1 prompt (the given prompt, as first sent)

The six-step specification was sent in the same message, directly above the prompt. The `<paste the six numbered steps here>` placeholder inside the prompt was left as it was.

**Specification sent with it:**

```
1. Fit the restricted model: your DiD regression with the treatment coefficient forced to zero. Keep the restricted residuals ũi and fitted values ỹi.
2. Draw one Rademacher weight per board — +1 or −1, each with probability one half. One weight per board, not per observation.
3. Build a synthetic outcome y*i = ỹi + wb(i) ũi, where b(i) is county i's board.
4. Re-estimate the unrestricted model on y* and store the cluster-robust t-statistic on the treatment coefficient.
5. The bootstrap p-value is the share of | t* | at least as large as your actual | t |.
6. With 8 boards there are only 2^8 = 256 distinct weight vectors, so enumerate all 256 rather than resampling. Say in a text cell why that makes the number of replications irrelevant here, and what it implies for the smallest p-value you are able to report.
```

**Prompt:**

```
[Prep] Act as an applied microeconometrician who has implemented
cluster-robust inference for panels with very few clusters.

[Request] Implement the wild cluster bootstrap exactly to the
specification below, for a county-year panel with treatment assigned
at the level of eight regional boards. Impose the null, apply
Rademacher weights at the cluster level, enumerate all 256 weight
vectors rather than resampling, and return the bootstrap p-value for
the treatment coefficient.
<paste the six numbered steps here>

[Iterate] numpy, pandas and statsmodels only. Reuse my panel's column
names. The function must accept the cluster variable as an argument.

[Mechanism Check] Explain why the weights are applied to residuals at
the CLUSTER level rather than the observation level, why imposing the
null matters, and what changes when the number of clusters drops
below roughly ten.

[Evaluate] Run it on my panel and compare the p-value against the
naive cluster-robust one. Tell me which conclusion changes.
```

## Version 1 reply (raw, unedited)

> I'll implement it to the six-step specification, check it against statsmodels, and run it on your panel.
>
> The function matches statsmodels' cluster-robust t to 12 decimal places. The agency's TWFE coefficient isn't significant under any method, so I'll also run the bootstrap on the corrected estimate from Step 3.3, where the choice of inference actually matters.
>
> ## The function
>
> This follows your six steps exactly. One generalisation: besides a single coefficient (`term=`), it also accepts a `contrast=` vector, so it can test the corrected estimate from Step 3.3, which is a weighted average of event-study lags rather than one coefficient. The null is imposed by restricted least squares, which for a single coefficient is the same as dropping that column.

```python
# ---- Task 4.1: Wild cluster bootstrap, enumerated ----
import itertools


def _cr1_stat(X: np.ndarray, y: np.ndarray, groups: np.ndarray,
              r: np.ndarray) -> tuple[float, float]:
    """Estimate of r'beta and its CR1 cluster-robust t-statistic (statsmodels' scaling)."""
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    u = y - X @ beta
    meat = np.zeros((X.shape[1], X.shape[1]))
    for g in np.unique(groups):
        s = X[groups == g].T @ u[groups == g]
        meat += np.outer(s, s)
    n, k, G = X.shape[0], np.linalg.matrix_rank(X), len(np.unique(groups))
    V = (G / (G - 1)) * ((n - 1) / (n - k)) * XtX_inv @ meat @ XtX_inv
    est = r @ beta
    return float(est), float(est / np.sqrt(r @ V @ r))


def wild_cluster_bootstrap(data: pd.DataFrame, formula: str, cluster: str,
                           term: str | None = None, contrast: np.ndarray | None = None) -> dict:
    """Wild cluster restricted bootstrap, Rademacher weights, all 2^G weight vectors enumerated.

    Tests H0: coefficient `term` = 0, or H0: contrast @ beta = 0 (e.g. a weighted average of
    event-study lags). `cluster` is the column treatment was assigned on.

    1. Fit the restricted model (null imposed); keep its fitted values and residuals.
    2. One Rademacher weight (+1/-1) per cluster -- every one of the 2^G vectors.
    3. y* = restricted fitted + w[cluster of row] * restricted residual.
    4. Re-fit the unrestricted model on y*, store the cluster-robust t.
    5. p-value = share of |t*| >= |t| on the real data.
    """
    model = smf.ols(formula, data=data)
    y, X = model.endog, model.exog
    if contrast is None:
        contrast = np.zeros(X.shape[1])
        contrast[model.exog_names.index(term)] = 1.0
    r = np.asarray(contrast, dtype=float)
    groups = data.loc[model.data.row_labels, cluster].to_numpy()

    est, t_hat = _cr1_stat(X, y, groups, r)

    # Step 1: restricted least squares with r'beta = 0 imposed
    XtX_inv = np.linalg.pinv(X.T @ X)
    beta = XtX_inv @ (X.T @ y)
    beta_r = beta - XtX_inv @ r * (r @ beta) / (r @ XtX_inv @ r)
    fitted_r = X @ beta_r
    resid_r = y - fitted_r

    # Steps 2-4: enumerate every Rademacher vector
    clusters, idx = np.unique(groups, return_inverse=True)
    vectors = np.array(list(itertools.product([-1.0, 1.0], repeat=len(clusters))))
    t_star = np.array([_cr1_stat(X, fitted_r + w[idx] * resid_r, groups, r)[1] for w in vectors])

    # Step 5. The all-(+1) vector rebuilds the real data and all-(-1) its mirror image, so two
    # draws tie |t| exactly; a small tolerance stops rounding error from dropping a tie.
    tol = 1e-9 * max(1.0, abs(t_hat))
    return {"estimate": est, "t": t_hat,
            "p_value": float(np.mean(np.abs(t_star) >= abs(t_hat) - tol)),
            "clusters": len(clusters), "replications": len(vectors),
            "smallest_possible_p": 2 / len(vectors), "t_star": t_star}
```

> **Run it on your panel.** This uses `panel` from Step 2.1, and `es`, `formula` and `contrast` from Steps 3.2–3.3:

```python
twfe_formula = "y ~ treated + C(county) + C(year)"
checks = [("Agency's TWFE", panel, twfe_formula, {"term": "treated"}),
          ("Corrected (event-study ATT)", es, formula, {"contrast": contrast})]

rows = []
for label, data, f, kw in checks:
    r = kw.get("contrast")
    if r is None:
        r = np.zeros(len(smf.ols(f, data=data).fit().params))
        r[smf.ols(f, data=data).exog_names.index("treated")] = 1.0
    by_county = smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["county"]})
    by_board  = smf.ols(f, data=data).fit(cov_type="cluster", cov_kwds={"groups": data["board"]})
    wcb = wild_cluster_bootstrap(data, f, "board", **kw)
    assert np.isclose(wcb["t"], float(np.squeeze(by_board.t_test(r).tvalue)))   # same t as statsmodels
    rows.append({"estimate": label, "coef": wcb["estimate"],
                 "p, county-clustered": float(np.squeeze(by_county.t_test(r).pvalue)),
                 "p, board-clustered":  float(np.squeeze(by_board.t_test(r).pvalue)),
                 "p, wild bootstrap":   wcb["p_value"], "replications": wcb["replications"]})
display(pd.DataFrame(rows).set_index("estimate").round(4))

# Sanity check: remove the true effect from the data; the corrected estimate should NOT reject
es_null = es.assign(y=es["y"] - es["true_effect"])
null = wild_cluster_bootstrap(es_null, formula, "board", contrast=contrast)
print(f"No-effect check: estimate {null['estimate']:+.3f}, wild bootstrap p = {null['p_value']:.3f}")
print(f"Smallest p-value possible with 8 boards: 2/256 = {2 / 256:.4f}")
```

> **Output**
>
> | estimate | coef | p, county-clustered | p, board-clustered | p, wild bootstrap |
> |---|---|---|---|---|
> | Agency's TWFE | −0.310 | 0.367 | 0.699 | **0.797** |
> | Corrected (event-study ATT) | +1.840 | <0.0001 | <0.0001 | **0.0078** |
>
> ```
> No-effect check: estimate -0.068, wild bootstrap p = 0.781
> Smallest p-value possible with 8 boards: 2/256 = 0.0078
> ```
>
> **Checks I ran before trusting it**
>
> - **The t-statistic is right.** My cluster-robust t on the real data matches statsmodels' board-clustered t to 12 decimal places (−0.3867 for TWFE, 13.276 for the corrected estimate).
> - **It doesn't reject when there's nothing there.** With the true effect removed from the data, the corrected estimate is −0.07 and the bootstrap gives p = 0.78.
> - **I caught and fixed a tie bug.** My first version gave p = 0.0039 (1/256) for the corrected estimate, which is impossible. The all-(+1) weight vector rebuilds the real data, and the all-(−1) vector its exact mirror image, so both produce |t\*| = |t|, and the p-value can never fall below 2/256. Rounding made one of the two ties compare as slightly smaller and drop out. The tolerance in step 5 fixes that.
>
> ## Mechanism check
>
> **Why weights go on clusters, not observations.** The point of the bootstrap is to reproduce how the data are correlated. Counties in a board share the same yearly shock, so their residuals move together. Flipping a whole board's residuals with one weight keeps that shared movement intact. Flipping each county-year separately would break it, making the synthetic data look like 480 independent observations, which is the same mistake the default standard errors make.
>
> **Why imposing the null matters.** The p-value asks how often a t this large shows up *if the true effect is zero*, so the synthetic data have to be generated from a world where it is zero. Building them from the restricted model does that. Using unrestricted residuals instead builds them around the estimated effect, which makes the test less accurate, especially with few clusters.
>
> **What changes below about ten clusters.** Cluster-robust standard errors are justified as the number of clusters grows large. With 8, the variance estimate rests on only 8 cluster-level sums and is too noisy, usually too small, so ordinary cluster-robust tests reject too often. The wild cluster bootstrap holds its size much better, but with G clusters and Rademacher weights it has only 2^G possible draws, so it can't produce very small p-values.
>
> ## Evaluation
>
> **Does the agency's conclusion survive? Yes.** Its TWFE estimate of −0.31 pp was not significant with county-clustered errors (p = 0.37), and with the correct inference it is even less significant (wild bootstrap p = 0.80). Correcting the inference doesn't overturn "we find no significant effect". But this check can't fix the real problem: as Phase 2 showed, the −0.31 itself is biased by the already-treated comparisons. Correct standard errors around a wrong number are still a wrong number.
>
> **Does the corrected finding survive? Also yes.** The corrected estimate of +1.84 pp is overwhelmingly significant with cluster-robust errors (t = 13.3). The wild bootstrap gives p = 0.0078, the smallest p-value possible with 8 boards: no weight vector other than the two mirror-image ties produced a |t\*| as large. The effect holds up under the corrected inference. What the bootstrap changes is how precise the claim can be: with 8 boards, "p < 0.01" is the strongest statement the data can support, not the "p < 0.0001" that conventional clustered errors suggest.
>
> ## Text cell: why enumeration makes replications irrelevant

```markdown
**Why 256 replications, and the smallest p-value.** With 8 boards and a weight of +1 or −1 for each, there are exactly 2⁸ = 256 possible weight vectors. Enumerating all of them gives the exact bootstrap distribution. Drawing at random, even 99,999 times, would only approximate the same 256 values with repeats, so the number of replications no longer matters: there is no simulation noise left. It also sets a floor on the p-value. The all-(+1) vector reproduces the real data and the all-(−1) vector its mirror image, so at least 2 of the 256 draws always match the observed |t|. The smallest p-value I can report is therefore **2/256 ≈ 0.0078**. The corrected estimate hits that floor, so the honest statement is "p < 0.01", not anything smaller.
```

---

## Version 2 prompt (my revision, as sent)

Sent as a follow-up in the same chat, after reading the version 1 reply:

```
Add the t(7) comparison. With 8 boards, a cluster-robust test should use t with 7 degrees of freedom, not the normal. Add: "Also report the board-clustered p-value using a t distribution with 7 degrees of freedom."
```

```python
    t_board = float(np.squeeze(by_board.t_test(r).tvalue))
    G = data["board"].nunique()
    rows.append({"estimate": label, "coef": wcb["estimate"],
                 "p, county-clustered":      float(np.squeeze(by_county.t_test(r).pvalue)),
                 "p, board-clustered (normal)": float(np.squeeze(by_board.t_test(r).pvalue)),
                 "p, board-clustered t(7)":  2 * stats.t.sf(abs(t_board), G - 1),
                 "p, wild bootstrap":        wcb["p_value"], "replications": wcb["replications"]})
```

> **Output**
>
> | estimate | coef | p, county-clustered | p, board (normal) | p, board t(7) | p, wild bootstrap |
> |---|---|---|---|---|---|
> | Agency's TWFE | −0.310 | 0.367 | 0.699 | 0.710 | **0.797** |
> | Corrected (event-study ATT) | +1.840 | <0.0001 | <0.0001 | <0.0001 (3.2 × 10⁻⁶) | **0.0078** |
>
> **What the t(7) comparison adds**
>
> - **Agency's TWFE:** switching from the normal to t(7) barely changes anything (0.699 → 0.710). The estimate is nowhere near significant either way, and the bootstrap (0.797) agrees.
> - **Corrected estimate:** t(7) has a big effect in absolute terms, moving the p-value from about 10⁻⁴⁰ to 3 × 10⁻⁶. Fat tails matter when t = 13.3 rests on only 8 boards. But it's still far below the bootstrap's 0.0078, so the t(7) correction is not enough on its own. Even with the right degrees of freedom, conventional clustered inference claims much more precision than 8 boards can support, and only the bootstrap reflects that limit.
> - **Conclusions don't change:** the agency's estimate is not significant under any method, and the corrected estimate is significant under every method. What changes is the strength of the claim: "p < 0.01", not "p < 0.000001".
