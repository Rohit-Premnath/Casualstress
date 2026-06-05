"""
compare_hmm_vs_gmm.py
======================
Implements Gaussian HMM from scratch (pure numpy — no hmmlearn, no file changes).
Runs it against the same 15 crisis events used in the GMM validation and
compares crisis detection accuracy side-by-side.

Run from repo root:
    python compare_hmm_vs_gmm.py
"""

import os
import sys
import numpy as np
import pandas as pd
from scipy.special import logsumexp
from sklearn.cluster import KMeans
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, os.path.dirname(__file__))
import psycopg2

# ── Database ──────────────────────────────────────────────────────────────────

def get_conn():
    return psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", "5433"),
        dbname=os.getenv("POSTGRES_DB", "causalstress"),
        user=os.getenv("POSTGRES_USER", "causalstress"),
        password=os.getenv("POSTGRES_PASSWORD", "causalstress_dev_2026"),
    )


def load_features():
    """Load the same 8 regime features from the processed data."""
    FEATURES = ["^VIX", "BAMLH0A0HYM2", "T10Y2Y", "^GSPC", "^MOVE",
                "TEDRATE", "STLFSI4"]

    conn = get_conn()
    df = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE variable_code = ANY(%s)
        ORDER BY date
    """, conn, params=(FEATURES,))

    # Also load realized vol (engineered feature)
    rvol = pd.read_sql("""
        SELECT date, variable_code, transformed_value
        FROM processed.time_series_data
        WHERE variable_code = '^GSPC_vol_21d'
        ORDER BY date
    """, conn)
    conn.close()

    pivoted = df.pivot_table(index="date", columns="variable_code", values="transformed_value")
    if not rvol.empty:
        rvol_piv = rvol.pivot_table(index="date", columns="variable_code", values="transformed_value")
        rvol_piv.columns = ["SPX_RVOL_21"]
        pivoted = pivoted.join(rvol_piv, how="left")

    pivoted.index = pd.to_datetime(pivoted.index)
    pivoted = pivoted.dropna()
    print(f"  Loaded {len(pivoted)} days x {len(pivoted.columns)} features")
    print(f"  Date range: {pivoted.index[0].date()} to {pivoted.index[-1].date()}")
    return pivoted


# ── Gaussian HMM (pure numpy) ─────────────────────────────────────────────────

class GaussianHMM_numpy:
    """
    Gaussian HMM with full covariance, implemented in log-space numpy.
    Baum-Welch EM for training, Viterbi for decoding.
    """

    def __init__(self, n_components=4, n_iter=200, tol=1e-3, random_state=42):
        self.K = n_components
        self.n_iter = n_iter
        self.tol = tol
        self.rng = np.random.default_rng(random_state)
        self.converged_ = False

    # ── Emission log-likelihood ───────────────────────────────────────────────

    def _log_emission(self, X):
        """Return (T, K) matrix of Gaussian log-likelihoods."""
        T, D = X.shape
        out = np.zeros((T, self.K))
        for k in range(self.K):
            diff = X - self.means_[k]
            try:
                L = np.linalg.cholesky(self.covs_[k])
                sol = np.linalg.solve(L, diff.T)            # (D, T)
                log_det = 2 * np.sum(np.log(np.diag(L)))
                maha = np.sum(sol ** 2, axis=0)             # (T,)
            except np.linalg.LinAlgError:
                cov = self.covs_[k] + np.eye(D) * 1e-6
                inv = np.linalg.inv(cov)
                maha = np.einsum("td,de,te->t", diff, inv, diff)
                sign, log_det = np.linalg.slogdet(cov)
                log_det = log_det if sign > 0 else 1e10
            out[:, k] = -0.5 * (D * np.log(2 * np.pi) + log_det + maha)
        return out

    # ── Forward algorithm (log-space) ─────────────────────────────────────────

    def _forward(self, log_e):
        T, K = log_e.shape
        log_alpha = np.full((T, K), -np.inf)
        log_alpha[0] = self.log_pi_ + log_e[0]
        for t in range(1, T):
            for k in range(K):
                log_alpha[t, k] = logsumexp(log_alpha[t - 1] + self.log_A_[:, k]) + log_e[t, k]
        return log_alpha

    # ── Backward algorithm (log-space) ────────────────────────────────────────

    def _backward(self, log_e):
        T, K = log_e.shape
        log_beta = np.zeros((T, K))          # log(1) = 0
        for t in range(T - 2, -1, -1):
            for k in range(K):
                log_beta[t, k] = logsumexp(
                    self.log_A_[k, :] + log_e[t + 1, :] + log_beta[t + 1, :]
                )
        return log_beta

    # ── Initialise parameters via k-means ────────────────────────────────────

    def _init_params(self, X):
        T, D = X.shape
        km = KMeans(n_clusters=self.K, n_init=5, random_state=int(self.rng.integers(1e6)))
        labels = km.fit_predict(X)

        self.means_ = km.cluster_centers_.copy()
        self.covs_ = np.array([
            np.cov(X[labels == k].T) + np.eye(D) * 1e-4
            if (labels == k).sum() > D
            else np.eye(D) * 0.1
            for k in range(self.K)
        ])

        # Uniform transition + slight self-persistence
        A = np.full((self.K, self.K), 0.1 / (self.K - 1))
        np.fill_diagonal(A, 0.9)
        self.log_A_ = np.log(A)
        self.log_pi_ = np.log(np.ones(self.K) / self.K)

    # ── Baum-Welch EM ─────────────────────────────────────────────────────────

    def fit(self, X):
        X = np.array(X, dtype=float)
        T, D = X.shape
        self._init_params(X)

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            log_e = self._log_emission(X)
            log_alpha = self._forward(log_e)
            log_beta  = self._backward(log_e)

            # Log-likelihood
            ll = logsumexp(log_alpha[-1])

            # Posterior state probabilities γ
            log_gamma = log_alpha + log_beta
            log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
            gamma = np.exp(log_gamma)                              # (T, K)

            # Pairwise posteriors ξ  (T-1, K, K)
            log_xi = np.full((T - 1, self.K, self.K), -np.inf)
            for t in range(T - 1):
                for j in range(self.K):
                    for k in range(self.K):
                        log_xi[t, j, k] = (log_alpha[t, j]
                                           + self.log_A_[j, k]
                                           + log_e[t + 1, k]
                                           + log_beta[t + 1, k])
                log_xi[t] -= logsumexp(log_xi[t])

            xi = np.exp(log_xi)                                    # (T-1, K, K)

            # ── M-step ───────────────────────────────────────────────────────
            # Initial state
            self.log_pi_ = np.log(gamma[0] + 1e-300)
            self.log_pi_ -= logsumexp(self.log_pi_)

            # Transition matrix
            new_A = xi.sum(axis=0) + 1e-10                        # (K, K)
            new_A /= new_A.sum(axis=1, keepdims=True)
            self.log_A_ = np.log(new_A)

            # Emission means & covariances
            g_sum = gamma.sum(axis=0)                              # (K,)
            for k in range(self.K):
                w = gamma[:, k]
                self.means_[k] = (w @ X) / (g_sum[k] + 1e-300)
                diff = X - self.means_[k]
                self.covs_[k] = (w[:, None] * diff).T @ diff / (g_sum[k] + 1e-300)
                self.covs_[k] += np.eye(D) * 1e-4                 # regularise

            if abs(ll - prev_ll) < self.tol:
                self.converged_ = True
                break
            prev_ll = ll

        self.log_likelihood_ = ll
        return self

    # ── Viterbi decoding ──────────────────────────────────────────────────────

    def predict(self, X):
        X = np.array(X, dtype=float)
        T = len(X)
        log_e = self._log_emission(X)
        delta = np.full((T, self.K), -np.inf)
        psi   = np.zeros((T, self.K), dtype=int)
        delta[0] = self.log_pi_ + log_e[0]
        for t in range(1, T):
            for k in range(self.K):
                vals = delta[t - 1] + self.log_A_[:, k]
                psi[t, k]   = np.argmax(vals)
                delta[t, k] = vals[psi[t, k]] + log_e[t, k]
        states = np.zeros(T, dtype=int)
        states[-1] = np.argmax(delta[-1])
        for t in range(T - 2, -1, -1):
            states[t] = psi[t + 1, states[t + 1]]
        return states

    def predict_proba(self, X):
        X = np.array(X, dtype=float)
        log_e     = self._log_emission(X)
        log_alpha = self._forward(log_e)
        log_beta  = self._backward(log_e)
        log_gamma = log_alpha + log_beta
        log_gamma -= logsumexp(log_gamma, axis=1, keepdims=True)
        return np.exp(log_gamma)

    def score(self, X):
        X = np.array(X, dtype=float)
        log_e = self._log_emission(X)
        log_alpha = self._forward(log_e)
        return logsumexp(log_alpha[-1]) / len(X)       # per-sample

    def bic(self, X):
        T, D = np.array(X).shape
        K = self.K
        n_params = K * (K - 1) + K * D + K * D * (D + 1) / 2
        return -2 * self.score(X) * T + n_params * np.log(T)


# ── Crisis event validation ───────────────────────────────────────────────────

CRISIS_PERIODS = [
    ("Global Financial Crisis",       "2008-09-01",  "2009-03-31"),
    ("Flash Crash / Euro Crisis",      "2010-05-01",  "2010-08-31"),
    ("US Debt Downgrade / Euro Debt",  "2011-07-01",  "2011-10-31"),
    ("China Devaluation / Oil Crash",  "2015-08-01",  "2016-02-29"),
    ("Volmageddon",                    "2018-01-26",  "2018-04-30"),
    ("Fed Tightening Selloff",         "2018-09-20",  "2018-12-31"),
    ("COVID Crash",                    "2020-02-19",  "2020-04-30"),
    ("Rate Hike Selloff",              "2021-12-01",  "2022-06-30"),
    ("US Debt Ceiling Crisis",         "2023-03-08",  "2023-05-31"),
    ("Brexit Shock",                   "2016-06-23",  "2016-07-15"),
    ("Oil Price War (COVID + OPEC)",   "2020-03-01",  "2020-04-30"),
    ("China Stock Market Crash",       "2015-06-12",  "2015-09-30"),
    ("October 2018 Correction",        "2018-10-01",  "2018-12-31"),
    ("September 2020 Tech Selloff",    "2020-09-02",  "2020-10-31"),
    ("SVB Banking Crisis",             "2023-03-08",  "2023-04-30"),
]

STRESS_REGIMES = {"elevated", "stressed", "high_stress", "crisis"}


def label_names(n_states, regime_labels, data, feature_cols):
    """Sort regimes by VIX mean and assign names."""
    vix_col = "^VIX" if "^VIX" in feature_cols else feature_cols[0]
    vix_means = {}
    for k in range(n_states):
        mask = regime_labels == k
        vix_means[k] = data.loc[mask, vix_col].mean() if mask.sum() > 0 else 0

    sorted_by_vix = sorted(vix_means, key=lambda k: vix_means[k])

    name_lists = {
        2: ["calm", "crisis"],
        3: ["calm", "stressed", "crisis"],
        4: ["calm", "normal", "stressed", "crisis"],
        5: ["calm", "normal", "elevated", "stressed", "crisis"],
        6: ["calm", "normal", "elevated", "stressed", "high_stress", "crisis"],
    }
    names = name_lists.get(n_states, [f"regime_{i}" for i in range(n_states)])
    return {k: names[i] for i, k in enumerate(sorted_by_vix)}


def validate_crises(regime_series, label):
    print(f"\n{'Crisis Period':<42} {'Detected As':<18} {'Match'}")
    print("  " + "-" * 72)
    correct = 0
    for name, start, end in CRISIS_PERIODS:
        mask = (regime_series.index >= start) & (regime_series.index <= end)
        period = regime_series[mask]
        if len(period) == 0:
            dominant = "no data"
            match = False
        else:
            dominant = period.mode().iloc[0]
            match = dominant in STRESS_REGIMES
        correct += int(match)
        tick = "YES" if match else " NO"
        print(f"  {name:<42} {dominant:<18} {tick}")

    pct = correct / len(CRISIS_PERIODS) * 100
    print(f"\n  [{label}] Crisis detection accuracy: {correct}/{len(CRISIS_PERIODS)} = {pct:.1f}%")
    return pct


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("  HMM vs GMM — Crisis Detection Comparison")
    print("  (HMM implemented in pure numpy — no hmmlearn needed)")
    print("=" * 70)

    print("\nLoading features from database...")
    raw = load_features()
    feature_cols = list(raw.columns)

    # Standardise
    means_ = raw.mean()
    stds_  = raw.std().replace(0, 1)
    X = ((raw - means_) / stds_).values

    # ── Select optimal n_states via BIC ──────────────────────────────────────
    print("\nSelecting optimal states for HMM (BIC, 2-6)...")
    best_bic, best_n, best_model = np.inf, 2, None

    for n in range(2, 7):
        print(f"  Fitting HMM with {n} states ...", end=" ", flush=True)
        try:
            model = GaussianHMM_numpy(n_components=n, n_iter=150, tol=1e-3, random_state=42)
            model.fit(X)
            b = model.bic(X)
            print(f"BIC={b:.1f}  converged={model.converged_}")
            if b < best_bic:
                best_bic, best_n, best_model = b, n, model
        except Exception as e:
            print(f"FAILED — {e}")

    print(f"\n  Optimal HMM states: {best_n}")

    # ── HMM regime labels ─────────────────────────────────────────────────────
    print("\nDecoding HMM regime labels (Viterbi)...")
    hmm_labels = best_model.predict(X)
    hmm_name_map = label_names(best_n, hmm_labels, raw, feature_cols)
    hmm_regime_series = pd.Series(
        [hmm_name_map[l] for l in hmm_labels], index=raw.index
    )

    print("\nHMM regime distribution:")
    for name, count in hmm_regime_series.value_counts().items():
        pct = count / len(hmm_regime_series) * 100
        print(f"  {name:<14} {count:>5} days  ({pct:.1f}%)")

    # ── Load GMM labels from database ────────────────────────────────────────
    print("\nLoading GMM regime labels from database...")
    conn = get_conn()
    gmm_df = pd.read_sql("""
        SELECT date, regime_name FROM models.regimes ORDER BY date
    """, conn)
    conn.close()
    gmm_df["date"] = pd.to_datetime(gmm_df["date"])
    gmm_regime_series = gmm_df.set_index("date")["regime_name"]

    print("GMM regime distribution:")
    for name, count in gmm_regime_series.value_counts().items():
        pct = count / len(gmm_regime_series) * 100
        print(f"  {name:<14} {count:>5} days  ({pct:.1f}%)")

    # ── Crisis validation ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  CRISIS DETECTION — HMM (pure numpy)")
    print("=" * 70)
    hmm_acc = validate_crises(hmm_regime_series, "HMM")

    print("\n" + "=" * 70)
    print("  CRISIS DETECTION — GMM (from database)")
    print("=" * 70)
    gmm_acc = validate_crises(gmm_regime_series, "GMM")

    # ── Side-by-side summary ──────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("  FINAL COMPARISON")
    print("=" * 70)
    print(f"  {'Method':<10} {'States':>7}  {'Crisis Accuracy':>16}")
    print("  " + "-" * 38)
    print(f"  {'HMM':<10} {best_n:>7}  {hmm_acc:>15.1f}%")
    print(f"  {'GMM':<10} {'6':>7}  {gmm_acc:>15.1f}%")
    winner = "HMM" if hmm_acc > gmm_acc else ("GMM" if gmm_acc > hmm_acc else "TIE")
    print(f"\n  Winner: {winner}")
    print("=" * 70)

    # ── Per-event comparison ──────────────────────────────────────────────────
    print("\n  PER-EVENT COMPARISON:")
    print(f"  {'Crisis Period':<42} {'HMM':<18} {'GMM':<18}")
    print("  " + "-" * 80)
    for name, start, end in CRISIS_PERIODS:
        mask_h = (hmm_regime_series.index >= start) & (hmm_regime_series.index <= end)
        mask_g = (gmm_regime_series.index >= start) & (gmm_regime_series.index <= end)
        h = hmm_regime_series[mask_h].mode().iloc[0] if mask_h.sum() > 0 else "no data"
        g = gmm_regime_series[mask_g].mode().iloc[0] if mask_g.sum() > 0 else "no data"
        h_tick = "✓" if h in STRESS_REGIMES else "✗"
        g_tick = "✓" if g in STRESS_REGIMES else "✗"
        print(f"  {name:<42} {h_tick} {h:<16} {g_tick} {g:<16}")


if __name__ == "__main__":
    main()
