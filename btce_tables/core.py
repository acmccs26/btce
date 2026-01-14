# Auto-extracted core from btce_acm_ccs_trimmed.ipynb (no top-level execution)
# Generated: 2026-01-14T18:17:41.590755Z

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from dataclasses import dataclass
from enum import Enum
from scipy.stats import beta
import random
import os
import torch

# ============================================================================
# 1. CONFIGURATION
# ============================================================================

class UserType(Enum):
    LOYAL = 0
    DISGRUNTLED = 1 
    MALICIOUS = 2
    NEGLIGENT = 3 

@dataclass
class ChannelMetadata:
    index: int
    name: str
    category: str
    severity: float
    ll_weight: float = 1.0
# Experiment Config
NUM_USERS = 1000
PCT_MAL = .0025    
HORIZON = 30
THRESHOLD = 0.9 
N_MEMBERS = 25    
#ft = 3  
SACC_GAMMA = 0.05 # Clipping radius
# =====================================================================
# Signal-generation parameters (Beta distributions)
# =====================================================================
NORMAL_ALPHA, NORMAL_BETA = 3.5, 6.5   # mean ≈ 0.35
RECON_ALPHA,  RECON_BETA  = 5.0, 5.0   # mean 0.50
EXFIL_ALPHA,  EXFIL_BETA  = 8.0, 2.0   # mean 0.80



# ---------------------------------------------------------------------
# Signal columns (used for all downstream baselines)
# ---------------------------------------------------------------------
# Original notebook used 4 channels: s_logon, s_file, s_role, s_exfil.
# We replace s_role with two channels: s_email and s_device.
SIG_COLS = ["s_logon", "s_file", "s_email", "s_device", "s_exfil"]

# If set, agents will sample signals from this action-conditioned pool CSV.
SIGNAL_POOLS = None          # dict[action] -> np.ndarray shape (n_rows, len(SIG_COLS))
SIGNAL_POOL_CSV_PATH = None
FIT_DBN_FROM_POOL = True   # if using SIGNAL_POOLS, fit per-channel Beta params from the pool


# ============================================================================
# 2. CAUSAL DBN ENGINE
# ============================================================================

class CausalDependencyDBN:
    """Lightweight channel-factorized DBN emission model.

    - Backward compatible: if params[type] has scalar keys {'a','b'}, uses them for all channels.
    - Calibrated mode: if params[type] is a dict keyed by channel names, uses per-channel Beta(a,b).

    Use fit_from_pools() to estimate per-channel Beta params from an action-conditioned signal pool.
    """

    def __init__(self, channels):
        self.channels = channels

        # Default (synthetic) scalar likelihood params (Alpha, Beta)
        self.params = {
            UserType.LOYAL:       {'a': 2.0,  'b': 15.0},
            UserType.NEGLIGENT:   {'a': 3.0,  'b': 10.0},
            UserType.DISGRUNTLED: {'a': 5.0,  'b': 8.0},
            UserType.MALICIOUS:   {'a': 12.0, 'b': 3.0},
        }

    @staticmethod
    def _beta_moments_to_ab(mean, var, min_ab=0.2):
        """Method-of-moments for Beta(a,b), with guards for degenerate / discrete-heavy data."""
        mean = float(np.clip(mean, 1e-6, 1.0 - 1e-6))
        var = float(max(var, 1e-8))
        # Compute t = (m(1-m)/v) - 1 ; if v too large, t can go negative.
        t = (mean * (1.0 - mean) / var) - 1.0
        if not np.isfinite(t) or t <= 0:
            # Fall back to a mildly-informative symmetric beta around mean
            # Keep shape parameters >= min_ab
            a = max(min_ab, mean * 2.0)
            b = max(min_ab, (1.0 - mean) * 2.0)
            return a, b
        a = max(min_ab, mean * t)
        b = max(min_ab, (1.0 - mean) * t)
        return a, b

    def fit_from_pools(self, signal_pools: dict, action_to_type=None, fit_eps=1e-4):
        """Fit per-channel Beta(a,b) parameters from an action-conditioned pool.

        signal_pools: dict[action -> np.ndarray] where each array is shape (n_rows, n_channels)
        action_to_type: mapping from action string to UserType. Defaults:
            NORMAL -> LOYAL
            RECON  -> MALICIOUS
            EXFIL  -> MALICIOUS

        Writes self.params[user_type] as dict[channel_name -> {'a':..,'b':..}] for the types that
        appear in the mapping. Other types remain scalar defaults (or can be derived later).
        """
        if action_to_type is None:
            action_to_type = {
                "NORMAL": UserType.LOYAL,
                "RECON": UserType.MALICIOUS,
                "EXFIL": UserType.MALICIOUS,
            }

        # Collect matrices per user type (concatenate actions that map to same type)
        mats_by_type = {}
        for act, ut in action_to_type.items():
            X = signal_pools.get(act)
            if X is None:
                raise ValueError(f"Missing pool for action={act} needed to fit DBN params.")
            mats_by_type.setdefault(ut, []).append(X)

        for ut, mats in mats_by_type.items():
            X = np.vstack(mats) if len(mats) > 1 else mats[0]
            if X.ndim != 2 or X.shape[1] != len(self.channels):
                raise ValueError(
                    f"Pool matrix for {ut} has shape {X.shape}, expected (*,{len(self.channels)})."
                )

            per_ch = {}
            # Fit each channel independently
            for name, meta in self.channels.items():
                col = X[:, meta.index].astype(float)
                col = np.clip(col, fit_eps, 1.0 - fit_eps)

                m = float(np.mean(col))
                v = float(np.var(col))
                a, b = self._beta_moments_to_ab(m, v)
                per_ch[name] = {'a': float(a), 'b': float(b)}

            self.params[ut] = per_ch

        # Optionally derive intermediate types if desired:
        # If MALICIOUS and LOYAL are per-channel dicts, we can interpolate NEGLIGENT/DISGRUNTLED.
        if (isinstance(self.params.get(UserType.LOYAL), dict) and 'a' not in self.params[UserType.LOYAL]
            and isinstance(self.params.get(UserType.MALICIOUS), dict) and 'a' not in self.params[UserType.MALICIOUS]):
            # Simple convex interpolation in mean-space would be more principled, but this keeps it minimal:
            # use weighted average of a,b to create intermediate shapes.
            def interp_params(w):
                out = {}
                for ch in self.channels.keys():
                    aL, bL = self.params[UserType.LOYAL][ch]['a'], self.params[UserType.LOYAL][ch]['b']
                    aM, bM = self.params[UserType.MALICIOUS][ch]['a'], self.params[UserType.MALICIOUS][ch]['b']
                    out[ch] = {'a': float((1-w)*aL + w*aM), 'b': float((1-w)*bL + w*bM)}
                return out
            self.params[UserType.NEGLIGENT] = interp_params(0.25)
            self.params[UserType.DISGRUNTLED] = interp_params(0.55)

    def compute_log_likelihood(self, signals, user_type):
        """Sum log Beta pdf across channels, weighted by ll_weight."""
        log_prob = 0.0
        p = self.params.get(user_type, self.params[UserType.LOYAL])

        for name, meta in self.channels.items():
            val = float(signals[meta.index])
            val = np.clip(val, 1e-4, 1.0 - 1e-4)

            # Backward compatible scalar params
            if isinstance(p, dict) and ('a' in p and 'b' in p):
                a, b = p['a'], p['b']
            else:
                # Per-channel params
                ch = p.get(name) if isinstance(p, dict) else None
                if ch is None:
                    # fall back to loyal scalar if missing
                    base = self.params.get(UserType.LOYAL, {'a': 2.0, 'b': 15.0})
                    if isinstance(base, dict) and ('a' in base and 'b' in base):
                        a, b = base['a'], base['b']
                    else:
                        # loyal itself is per-channel
                        a, b = self.params[UserType.LOYAL][name]['a'], self.params[UserType.LOYAL][name]['b']
                else:
                    a, b = ch['a'], ch['b']

            log_prob += meta.ll_weight * np.log(beta.pdf(val, a, b) + 1e-12)

        return float(log_prob)

    def compute_threat_score(self, signals):
        """Weighted severity score with boosts for high values (used as ITS)."""
        score, weights = 0.0, 0.0
        for name, meta in self.channels.items():
            val = float(signals[meta.index])
            w = float(meta.severity)
            if val > 0.6:
                w *= 1.5
            if val > 0.8:
                w *= 2.0
            score += val * w
            weights += w
        return float(score / weights) if weights > 0 else 0.0

    # ------------------------------------------------------------------------
    # Exact pool-row caching (NO approximation)
    # ------------------------------------------------------------------------
    def attach_pool_cache(self, signal_pools: dict):
        """Attach an action-conditioned pool cache.

        This allocates per-action arrays for:
          - ll_loy (LOYAL log-likelihood)
          - ll_mal (MALICIOUS log-likelihood)
          - its    (threat score)
        Entries are filled lazily on first use to avoid expensive full precompute.
        """
        self._pool_cache = {}
        for act, X in (signal_pools or {}).items():
            n = int(X.shape[0])
            self._pool_cache[str(act)] = {
                "ll_loy": np.full(n, np.nan, dtype=np.float64),
                "ll_mal": np.full(n, np.nan, dtype=np.float64),
                "its":    np.full(n, np.nan, dtype=np.float64),
            }

    def reset_pool_cache(self):
        """Drop the attached pool cache (safe; falls back to direct computation)."""
        self._pool_cache = None

    def ll_cached(self, signals, user_type, pool_action=None, pool_idx=None):
        """Exact log-likelihood with optional pool-row caching."""
        if pool_action is not None and pool_idx is not None:
            cache = getattr(self, "_pool_cache", None)
            if cache is not None:
                act = str(pool_action)
                if act in cache:
                    arr = cache[act]["ll_mal"] if user_type == UserType.MALICIOUS else cache[act]["ll_loy"]
                    i = int(pool_idx)
                    v = arr[i]
                    if not np.isnan(v):
                        return float(v)
                    v = float(self.compute_log_likelihood(signals, user_type))
                    arr[i] = v
                    return v
        return float(self.compute_log_likelihood(signals, user_type))

    def its_cached(self, signals, pool_action=None, pool_idx=None):
        """Exact ITS with optional pool-row caching."""
        if pool_action is not None and pool_idx is not None:
            cache = getattr(self, "_pool_cache", None)
            if cache is not None:
                act = str(pool_action)
                if act in cache:
                    arr = cache[act]["its"]
                    i = int(pool_idx)
                    v = arr[i]
                    if not np.isnan(v):
                        return float(v)
                    v = float(self.compute_threat_score(signals))
                    arr[i] = v
                    return v
        return float(self.compute_threat_score(signals))



# ============================================================
# Ablation-ready overrides: UserAgent + ByzantineCommittee
# ============================================================

import numpy as np
import random

# Global toggles (default matches our original behavior)
BEHAVIOR_MODE = "behavioral"   # 'behavioral' or 'rational'
AGG_MODE      = "anchor_clip"  # 'anchor_clip', 'average', 'median'


class UserAgent:
    def __init__(self, uid, is_malicious_scenario):
        self.uid = uid
        self.is_target = is_malicious_scenario
        self.type = UserType.LOYAL
        self.time = 0

        # Stochastic start / peak times for attack progression
        self.start_day = np.random.randint(5, 15) if is_malicious_scenario else 999
        self.peak_day  = self.start_day + np.random.randint(8, 15)

        # Behavioral state (continuous “mood” variables)
        self.stress = 0.0
        self.ref_point = 0.0
        self.beta = 0.95

        # Intrinsic payoff parameters (match our original theta)
        self.theta = {
            "trust_utility":  2.0,
            "attack_utility": 25.0,
            "detection_cost": 8.0,
        }

        self.current_action = "NORMAL"
        self.cumulative_llr = 0.0

    def update_state(self, t):
        self.time = t
        if not self.is_target:
            return

        # ----- Rational benchmark: freeze behavioral dynamics -----
        if BEHAVIOR_MODE == "rational":
            self.type = UserType.MALICIOUS
            self.stress = 1.0
            self.ref_point = 0.0    # neutral reference
            self.beta = 1.0         # no extra "temperature"
            return

        # ----- Original behavioral evolution -----
        if t < self.start_day:
            self.type = UserType.LOYAL

        elif self.start_day <= t < self.peak_day:
            self.type = UserType.DISGRUNTLED
            progress = (t - self.start_day) / (self.peak_day - self.start_day)
            self.stress = progress
            self.ref_point = 3.0 * progress
            self.beta = 0.95 - (0.4 * progress)

        else:
            self.type = UserType.MALICIOUS
            self.stress = 1.0
            self.ref_point = 5.0
            self.beta = 0.5

    def choose_action(self, current_belief):
        # Loyal users behave normally
        if self.type == UserType.LOYAL:
            self.current_action = "NORMAL"
            return self.current_action

        # Disgruntled users: noisy recon vs normal
        if self.type == UserType.DISGRUNTLED:
            prob_recon = 0.1 + 0.5 * self.stress
            self.current_action = (
                "RECON" if np.random.random() < prob_recon else "NORMAL"
            )
            return self.current_action

        # Malicious regime: trade off attack vs detection
        p_mal = current_belief
        risk_tolerance = 0.2 + 0.1 * self.ref_point

        risk_term = self.theta["detection_cost"] * p_mal * (1.0 / risk_tolerance)
        u_attack = self.theta["attack_utility"] - risk_term
        u_normal = self.theta["trust_utility"]

        # ----- Rational mode: deterministic best reply -----
        if BEHAVIOR_MODE == "rational":
            self.current_action = "EXFIL" if u_attack >= u_normal else "RECON"
            return self.current_action

        # ----- Behavioral mode: softmax with inverse temperature 1/beta -----
        vals = np.array([u_normal, u_attack])
        inv_temp = 1.0 / self.beta
        probs = np.exp(vals * inv_temp) / np.sum(np.exp(vals * inv_temp))

        self.current_action = (
            "EXFIL" if np.random.random() < probs[1] else "RECON"
        )
        return self.current_action

    """def generate_signals(self, action):
        num_channels = 4
        # HIGHLY NOISY
        if action == "NORMAL":
            # mean ~ <0.45
            return np.random.beta(3.5, 6.5, size=num_channels)

        if action == "RECON":
            # mean ~ 0.50
            return np.random.beta(5, 5, size=num_channels)

        if action == "EXFIL":
            # mean ~ >0.55
            return np.random.beta(8, 2, size=num_channels)
        return np.random.beta(5, 5, size=num_channels)
    """


    def generate_signals(self, action, return_index: bool = False):
        """Emit signals conditioned on the chosen action.

        - If SIGNAL_POOLS is set, sample a row from the external pool for that action.
        - Otherwise fall back to the synthetic Beta generator.

        If return_index is True, returns (signals, pool_idx) where pool_idx is the row index
        inside SIGNAL_POOLS[action] (or None for synthetic fallback). This enables exact caching
        of likelihood terms with ZERO approximation.
        """
        global SIGNAL_POOLS

        if SIGNAL_POOLS is not None:
            pool = SIGNAL_POOLS[action]
            j = np.random.randint(0, pool.shape[0])
            sig = pool[j].copy()
            if return_index:
                return sig, int(j)
            return sig

        # synthetic fallback (4 base draws)
        if action == "NORMAL":
            base = np.random.beta(NORMAL_ALPHA, NORMAL_BETA, size=4)
        elif action == "RECON":
            base = np.random.beta(RECON_ALPHA, RECON_BETA, size=4)
        elif action == "EXFIL":
            base = np.random.beta(EXFIL_ALPHA, EXFIL_BETA, size=4)
        else:
            base = np.random.beta(RECON_ALPHA, RECON_BETA, size=4)

        s_logon, s_file, s_role_like, s_exfil = base
        sig = np.array([s_logon, s_file, s_role_like, s_role_like, s_exfil], dtype=float)
        if return_index:
            return sig, None
        return sig

class ByzantineCommittee:
    """
    Same structure as our original, but the aggregation step depends on AGG_MODE:
      - 'anchor_clip': clipped SACC (our baseline)
      - 'average':     naive mean of reports
      - 'median':      coordinate-wise median of reports
    """

    def __init__(self, channels):
        self.dbn = CausalDependencyDBN(channels)
        self.user_beliefs = {}
        self.naive_beliefs = {}
        self.belief_history = {}

    def initialize_user(self, uid):
        self.user_beliefs[uid] = 0.01
        self.naive_beliefs[uid] = 0.01
        self.belief_history[uid] = {
            "anchor": [],
            "sacc": [],
            "naive_smooth": [],
            "traitor_vote": [],
            "honest_votes": [],
        }

    def certify(self, uid, signals, ft, is_malicious: bool = False, pool_action=None, pool_idx=None, ll_m=None, ll_l=None):
        if uid not in self.user_beliefs:
            self.initialize_user(uid)

        prior = self.user_beliefs[uid]
        naive_prior = self.naive_beliefs[uid]

        # 1. Anchor update (Bayes with true likelihoods)
        if ll_m is None:
            ll_m = self.dbn.ll_cached(signals, UserType.MALICIOUS, pool_action=pool_action, pool_idx=pool_idx)
        if ll_l is None:
            ll_l = self.dbn.ll_cached(signals, UserType.LOYAL, pool_action=pool_action, pool_idx=pool_idx)
        likelihood_ratio = np.exp(ll_m - ll_l)
        anchor_post = (prior * likelihood_ratio) / (
            (prior * likelihood_ratio) + (1 - prior)
        )

        # 2. Committee voting: honest votes around anchor + Byzantine under-reporting
        reports = []
        honest_reports = []
        HONEST_NOISE_BASE   = 0.005  # noise when anchor is near 0
        HONEST_NOISE_SLOPE  = 0.02   # extra noise as belief rises
        LOYAL_NOISE_CUTOFF  = 0.65    # don't add extra noise below this

        for _ in range(N_MEMBERS - ft):
            # extra noise kicks in only when anchor > 0.5
            extra = max(anchor_post - LOYAL_NOISE_CUTOFF, 0.0)
            sigma = HONEST_NOISE_BASE + HONEST_NOISE_SLOPE * extra
            vote = np.clip(anchor_post + np.random.normal(0, sigma), 0.01, 0.99)
            reports.append(vote)
            honest_reports.append(vote)

        # --- traitor votes: mimic when anchor is low, sabotage when anchor is high ---
        risk_band_low  = THRESHOLD - SACC_GAMMA   # “getting close” to threshold
        #risk_band_high = THRESHOLD + SACC_GAMMA   # clearly in high-risk regime

        traitor_votes = []

        for _ in range(ft):
            if not is_malicious:
                # LOYAL USER: traitors blend in and look honest
                vote = np.clip(anchor_post + np.random.normal(0, 0.02), 0.01, 0.99)

            else:
                # MALICIOUS USER: traitors try to hide the attack
                if anchor_post < risk_band_low:
                    # Early malicious / low anchor: still blend in
                    vote = np.clip(anchor_post + np.random.normal(0, 0.02), 0.01, 0.99)
                else:
                    # Near or above threshold: aggressively under-report
                    # SACC will clip this to (anchor_post - SACC_GAMMA),
                    # maximizing downward pull inside the allowed band.
                    vote = anchor_post - 10 * SACC_GAMMA

            reports.append(vote)
            traitor_votes.append(vote)

        traitor_vote = float(np.mean(traitor_votes)) if traitor_votes else float("nan")
        raw_naive_mean = float(np.mean(reports))

        # 3. Aggregation rule selected by AGG_MODE
        if AGG_MODE == "anchor_clip":
            clipped_reports = []
            for r in reports:
                if abs(r - anchor_post) > SACC_GAMMA:
                    if r < anchor_post:
                        clipped_reports.append(anchor_post - SACC_GAMMA)
                    else:
                        clipped_reports.append(anchor_post + SACC_GAMMA)
                else:
                    clipped_reports.append(r)
            core_mean = float(np.mean(clipped_reports))

        elif AGG_MODE == "median":
            core_mean = float(np.median(reports))

        elif AGG_MODE == "average":
            core_mean = raw_naive_mean

        else:
            # Fallback to original clipped mean
            clipped_reports = []
            for r in reports:
                if abs(r - anchor_post) > SACC_GAMMA:
                    if r < anchor_post:
                        clipped_reports.append(anchor_post - SACC_GAMMA)
                    else:
                        clipped_reports.append(anchor_post + SACC_GAMMA)
                else:
                    clipped_reports.append(r)
            core_mean = float(np.mean(clipped_reports))

        # 4. Inertia / smoothing
        alpha = 0.4
        system_belief = (1 - alpha) * prior + alpha * core_mean
        naive_belief_smooth = (1 - alpha) * naive_prior + alpha * raw_naive_mean

        self.user_beliefs[uid] = system_belief
        self.naive_beliefs[uid] = naive_belief_smooth

        # 5. Log histories for later analysis
        self.belief_history[uid]["anchor"].append(anchor_post)
        self.belief_history[uid]["sacc"].append(system_belief)
        self.belief_history[uid]["naive_smooth"].append(naive_belief_smooth)
        self.belief_history[uid]["traitor_vote"].append(traitor_vote)
        self.belief_history[uid]["honest_votes"].append(honest_reports)

        return system_belief




# ============================================================================
# 7. MAIN
# ============================================================================


# ============================================================================
# 5. SIMULATION LOOP
# ============================================================================

def run_simulation(ft):
    channels = {
        's_logon':  ChannelMetadata(0, 's_logon',  'Access', 0.6, ll_weight=1.0),
        's_file':   ChannelMetadata(1, 's_file',   'Recon',  0.7, ll_weight=1.0),
        # role -> (email, device) split; weights sum to the former single channel weight.
        's_email':  ChannelMetadata(2, 's_email',  'Priv',   0.45, ll_weight=0.5),
        's_device': ChannelMetadata(3, 's_device', 'Priv',   0.45, ll_weight=0.5),
        's_exfil':  ChannelMetadata(4, 's_exfil',  'Exfil',  0.9, ll_weight=1.0),
    }
    
    committee = ByzantineCommittee(channels)
    # If using an external action-conditioned signal pool, calibrate the DBN emission params
    # from the pool to avoid miscalibration on sparse CERT-like signals.
    global SIGNAL_POOLS, FIT_DBN_FROM_POOL
    if (SIGNAL_POOLS is not None) and FIT_DBN_FROM_POOL:
        committee.dbn.fit_from_pools(SIGNAL_POOLS)

    # Attach exact per-pool-row caches (lazy fill, no approximation)
    if SIGNAL_POOLS is not None:
        committee.dbn.attach_pool_cache(SIGNAL_POOLS)

    logs = []
    n_mal = int(NUM_USERS * PCT_MAL)
    
    agents = [UserAgent(i, is_malicious_scenario=(i < n_mal)) for i in range(NUM_USERS)]

    print(f"Running v21 Sim (Annotated CUSUM): {NUM_USERS} Users, {HORIZON} Steps")

    for t in range(HORIZON): # every day
        for uid in range(NUM_USERS): # each user is up to something
            agent = agents[uid]
            agent.update_state(t)
            
            # Action & Signals
            current_belief = committee.user_beliefs.get(uid, 0.01)
            action = agent.choose_action(current_belief)
            signals, pool_idx = agent.generate_signals(action, return_index=True)
            
            # CUSUM Logic
            ll_mal = committee.dbn.ll_cached(signals, UserType.MALICIOUS, pool_action=action, pool_idx=pool_idx)
            ll_loy = committee.dbn.ll_cached(signals, UserType.LOYAL, pool_action=action, pool_idx=pool_idx)
            llr = ll_mal - ll_loy
            agent.cumulative_llr += llr 

            # Ground-truth malicious flag for this period
            is_malicious = (agent.type == UserType.MALICIOUS)
            
            p_mal = committee.certify(uid, signals, ft, is_malicious=is_malicious, pool_action=action, pool_idx=pool_idx, ll_m=ll_mal, ll_l=ll_loy)
            its = committee.dbn.its_cached(signals, pool_action=action, pool_idx=pool_idx)
            hist = committee.belief_history[uid]

            s_logon, s_file, s_email, s_device, s_exfil = signals

            # Helper to get mean of honest votes for logging/viz if needed
            honest_mean = np.mean(hist['honest_votes'][-1]) if hist['honest_votes'] else 0.0
            
            logs.append({
                'Day': t,
                'AgentID': uid,
                'Type': agent.type.name,
                'Action': action,
                'ITS': its,
                'RefPoint': agent.ref_point,
                'Beta': agent.beta,
                'P_Detect': p_mal,
                'Naive_Smooth': hist['naive_smooth'][-1],
                'IsTarget': agent.is_target,
                'Signals_Mean': np.mean(signals),
                'Traitor_Vote': hist['traitor_vote'][-1],
                'CUSUM': agent.cumulative_llr,

                # ---- raw observed signals (WHAT baselines should see) ----
                "s_logon": float(s_logon),
                "s_file": float(s_file),
                "s_email": float(s_email),
                "s_device": float(s_device),
                "s_exfil": float(s_exfil),
                
                'Honest_Mean': honest_mean
            })
            
    return pd.DataFrame(logs)

import os, re, math, random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score


# ----------------------------
# 0) Repro + utilities
# ----------------------------
def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_iterable_thresholds(thresholds):
    if isinstance(thresholds, (int, float, np.integer, np.floating)):
        return (float(thresholds),)
    return tuple(thresholds)

def infer_transformer_feature_cols(df, *, drop_cols=None):
    """
    Transformer features = all s_* columns EXCEPT:
      - s_exfil  (BTCE construct / too direct)
      - anything containing 'killchain' (BTCE construct)
    """
    if drop_cols is None:
        drop_cols = set()

    cols = []
    for c in df.columns:
        cl = c.lower()
        if not c.startswith("s_"):
            continue
        if cl == "s_exfil":
            continue
        if "killchain" in cl:
            continue
        if c in drop_cols:
            continue
        cols.append(c)

    if len(cols) == 0:
        raise ValueError("No transformer feature columns found after exclusions.")
    return cols



def add_killchain_labels(
    df: pd.DataFrame,
    user_col: str = "AgentID",
    time_col: str = "Day",
    action_col: str = "Action",
    recon_value: str = "RECON",
    exfil_value: str = "EXFIL",
) -> pd.DataFrame:
    """
    Adds:
      - y_exfil: 1 on EXFIL periods
      - y_killchain: 1 on (RECON periods that occur on/before first EXFIL) + all EXFIL periods
                     (prevents baselines from getting credit for RECON unless EXFIL occurs later)
      - y_attack_anyexfil: 1 on any RECON or EXFIL period for target users who EXFIL at least once
                           (rational-friendly: no RECON->EXFIL ordering requirement)
    """
    out = df.sort_values([user_col, time_col]).copy()

    out["y_exfil"] = (out[action_col].astype(str) == exfil_value).astype(int)

    # first EXFIL time per user (NaN if none)
    t_exfil_start = out.loc[out["y_exfil"] == 1].groupby(user_col)[time_col].min()
    out["_t_exfil_start"] = out[user_col].map(t_exfil_start)

    is_target = (out["IsTarget"].astype(int) == 1)
    has_exfil = out["_t_exfil_start"].notna()
    is_recon  = (out[action_col].astype(str) == recon_value)
    is_exfil  = (out[action_col].astype(str) == exfil_value)

    # killchain: RECON only counts if on/before first EXFIL; EXFIL always counts
    recon_before_exfil = is_recon & (out[time_col] <= out["_t_exfil_start"])
    out["y_killchain"] = (is_target & has_exfil & (recon_before_exfil | is_exfil)).astype(int)

    # rational-friendly: if EXFIL ever happens for the target user, count ALL RECON/EXFIL periods
    out["y_attack_anyexfil"] = (is_target & has_exfil & (is_recon | is_exfil)).astype(int)

    out.drop(columns=["_t_exfil_start"], inplace=True)
    return out

# ----------------------------
# 1) Transformer-UBS (factorized token model, NO leakage, CAUSAL)
# ----------------------------
class TokenSeqDataset(Dataset):
    """
    Builds per-user token sequences from continuous features in [0,1].
    Returns: (Tok[T,C], Y[T], meta=(AgentID, Days[T])).
    - label_mode="period": Y comes from df[label_col] per timestep (original behavior).
    - label_mode="user":   Y is constant per timestep, equal to user's IsTarget (or user_label_col).
    Optional prefix_train truncates each user sequence to a fixed prefix length (sampled once in __init__).
    """

    def __init__(self,
                 df: pd.DataFrame,
                 users,
                 feat_cols,
                 n_bins: int,
                 label_col: str = "",
                 poison_users=None,
                 *,
                 label_mode: str = "period",      # "period" or "user"
                 user_label_col: str = "IsTarget",
                 prefix_train: bool = False,
                 min_prefix: int = 3,
                 seed: int = 0):
        self.df = df[df["AgentID"].isin(users)].copy()
        self.feat_cols = list(feat_cols)
        self.n_bins = int(n_bins)
        self.label_col = str(label_col) if label_col is not None else ""
        self.label_mode = str(label_mode)
        self.user_label_col = str(user_label_col)
        self.prefix_train = bool(prefix_train)
        self.min_prefix = int(min_prefix)
        self.seed = int(seed)

        # Poison set (keep type as-is; supports str or int AgentID)
        if poison_users is None:
            self.poison_users = set()
        else:
            self.poison_users = set([u.item() if hasattr(u, "item") else u for u in poison_users])

        # Validate required columns
        required = {"AgentID", "Day"} | set(self.feat_cols)
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"TokenSeqDataset: missing required columns: {sorted(missing)}")

        if self.label_mode not in ("period", "user"):
            raise ValueError("label_mode must be 'period' or 'user'")

        if self.label_mode == "period":
            if self.label_col == "" or self.label_col not in self.df.columns:
                raise ValueError(f"label_mode='period' requires label_col present in df. Got label_col='{self.label_col}'")

        if self.label_mode == "user":
            if self.user_label_col not in self.df.columns:
                raise ValueError(f"label_mode='user' requires user_label_col present in df. Got '{self.user_label_col}'")

        # Precompute user-level labels (robust to repeats)
        if self.label_mode == "user":
            self.user_y = (self.df.groupby("AgentID")[self.user_label_col].max().astype(int).to_dict())
        else:
            self.user_y = {}

        # Build sequences
        self.tok, self.y, self.meta = [], [], []
        self.prefix_len = []

        rng = np.random.default_rng(self.seed)

        for uid, du in self.df.groupby("AgentID", sort=False):
            du = du.sort_values("Day")

            # Features -> tokens
            x = du[self.feat_cols].values.astype(np.float32)
            # Safety: clip to [0,1]
            x = np.clip(x, 0.0, 1.0)

            tok = np.floor(x * self.n_bins).astype(np.int64)
            tok = np.clip(tok, 0, self.n_bins - 1)

            # Labels
            if self.label_mode == "period":
                y = du[self.label_col].values.astype(np.float32)
                if uid in self.poison_users:
                    y = 1.0 - y
            else:
                y_u = float(self.user_y.get(uid, 0))
                if uid in self.poison_users:
                    y_u = 1.0 - y_u
                y = np.full(shape=(len(du),), fill_value=y_u, dtype=np.float32)

            days = du["Day"].values.astype(int)

            self.tok.append(torch.from_numpy(tok))
            self.y.append(torch.from_numpy(y))
            self.meta.append((uid, days))

            # Optional fixed prefix per user (sampled once here)
            if self.prefix_train:
                T = len(days)
                if T <= 0:
                    L = 0
                elif T <= self.min_prefix:
                    L = T
                else:
                    L = int(rng.integers(self.min_prefix, T + 1))
                self.prefix_len.append(L)
            else:
                self.prefix_len.append(None)

    def __len__(self):
        return len(self.tok)

    def __getitem__(self, i):
        Tok = self.tok[i]
        Y = self.y[i]
        uid, days = self.meta[i]

        L = self.prefix_len[i]
        if L is not None:
            Tok = Tok[:L]
            Y = Y[:L]
            days = days[:L]

        return Tok, Y, (uid, days)


def pad_collate_tok(batch):
    toks, ys, metas = zip(*batch)
    lens = torch.tensor([t.shape[0] for t in toks], dtype=torch.long)
    T = int(lens.max()); K = toks[0].shape[1]
    Tok = torch.zeros(len(toks), T, K, dtype=torch.long)
    Y = torch.zeros(len(toks), T)
    M = torch.zeros(len(toks), T, dtype=torch.bool)
    for i,(t,y) in enumerate(zip(toks,ys)):
        tt = t.shape[0]
        Tok[i,:tt] = t
        Y[i,:tt] = y
        M[i,:tt] = True
    return Tok, Y, M, metas

class PosEnc(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:,0::2] = torch.sin(pos*div)
        pe[:,1::2] = torch.cos(pos*div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x): 
        return x + self.pe[:, :x.size(1), :]

import inspect
import torch
import torch.nn as nn

class TransformerUBS(nn.Module):
    """
    Factorized tokens: one embedding table per channel, summed -> TransformerEncoder.
    CAUSAL mask ensures no future leakage (online-compatible scoring).
    Compatible across PyTorch versions where the encoder mask kwarg is either
    'mask' or 'src_mask'.
    """
    def __init__(self, n_bins: int, n_channels: int, d_model=128, nhead=4, nlayers=3, dropout=0.15):
        super().__init__()
        self.embs = nn.ModuleList([nn.Embedding(n_bins, d_model) for _ in range(n_channels)])
        self.pos = PosEnc(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, batch_first=True, dropout=dropout
        )
        self.enc = nn.TransformerEncoder(enc_layer, num_layers=nlayers)
        self.head = nn.Linear(d_model, 1)

        # cache for causal masks keyed by (T, device.type, device.index)
        self._causal_cache = {}

        # detect which kwarg name this torch version supports: 'src_mask' vs 'mask'
        params = inspect.signature(self.enc.forward).parameters
        self._mask_kw = "src_mask" if "src_mask" in params else "mask"

    def _get_causal_mask(self, T: int, device):
        key = (T, device.type, device.index)
        m = self._causal_cache.get(key)
        if m is None:
            # True = masked out (cannot attend)
            m = torch.triu(torch.ones(T, T, device=device, dtype=torch.bool), diagonal=1)
            self._causal_cache[key] = m
        return m

    def forward(self, Tok, mask):
        """
        Tok:  [B,T,K] int64 tokens
        mask: [B,T] bool, True where valid (not padding)
        returns logits [B,T]
        """
        B, T, K = Tok.shape

        h = 0
        for k in range(K):
            h = h + self.embs[k](Tok[:, :, k])
        h = self.pos(h)

        causal = self._get_causal_mask(T, Tok.device)   # [T,T] bool
        pad_mask = ~mask                                 # [B,T] bool

        if self._mask_kw == "src_mask":
            z = self.enc(h, src_mask=causal, src_key_padding_mask=pad_mask)
        else:
            z = self.enc(h, mask=causal, src_key_padding_mask=pad_mask)

        return self.head(z).squeeze(-1)


def masked_bce(logits, y, mask, pos_weight):
    logits = logits[mask]; y = y[mask]
    return nn.functional.binary_cross_entropy_with_logits(logits, y, pos_weight=pos_weight)

def train_transformer_ubs(df, feat_cols, train_users, val_users=None, *,
                          n_bins=8, epochs=20, batch=64, lr=3e-4, wd=1e-4,
                          poison_rate=0.0, seed=0, device=None,
                          label_col="y_killchain",
                          label_mode="period",          # NEW: "period" or "user"
                          user_label_col="IsTarget",    # NEW
                          prefix_train=False,           # NEW
                          min_prefix=3):                # NEW
    set_seed(seed)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    poison_users = None
    if poison_rate > 0:
        rng = np.random.default_rng(seed)
        poison_users = rng.choice(train_users, size=int(len(train_users)*poison_rate), replace=False)

    tr_ds = TokenSeqDataset(
        df, train_users, feat_cols,
        n_bins=n_bins,
        label_col=label_col,                 # used only if label_mode="period"
        label_mode=label_mode,               # NEW
        user_label_col=user_label_col,       # NEW
        poison_users=poison_users,
        prefix_train=prefix_train,           # NEW
        min_prefix=min_prefix,               # NEW
        seed=seed                             # NEW (for prefix rng)
    )
    tr_ld = DataLoader(tr_ds, batch_size=batch, shuffle=True, collate_fn=pad_collate_tok)

    model = TransformerUBS(n_bins=n_bins, n_channels=len(feat_cols)).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)

    # --- pos_weight ---
    # For user-label mode: compute from USERS (not timesteps) to avoid length bias.
    if label_mode == "user":
        y_user = (df.groupby("AgentID")[user_label_col].max().astype(int))
        y_tr = np.array([int(y_user.get(u, 0)) for u in train_users], dtype=float)
        pos = y_tr.sum()
        neg = len(y_tr) - pos
    else:
        # period-label mode: compute from masked timesteps ( original behavior)
        ys = []
        for Tok, Y, M, _ in tr_ld:
            ys.append(Y[M].numpy())
        ys = np.concatenate(ys) if len(ys) else np.array([0.0])
        pos = ys.sum()
        neg = len(ys) - pos

    pos_weight = torch.tensor([neg / (pos + 1e-12)], dtype=torch.float32, device=device)

    for ep in range(1, epochs + 1):
        model.train()
        for Tok, Y, M, _ in tr_ld:
            Tok, Y, M = Tok.to(device), Y.to(device), M.to(device)
            logits = model(Tok, M)
            loss = masked_bce(logits, Y, M, pos_weight)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()

    return model


@torch.no_grad()
def score_transformer(df, feat_cols, users, model,
                      n_bins=8, batch=64, device=None,
                      label_mode="period",
                      label_col="y_stage",
                      user_label_col="IsTarget"):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    ds = TokenSeqDataset(
        df, users, feat_cols, n_bins=n_bins,
        label_col=label_col,
        poison_users=None,
        label_mode=label_mode,          # <-- NEW
        user_label_col=user_label_col,  # <-- NEW
        prefix_train=False              # scoring uses full sequence
    )

    ld = DataLoader(ds, batch_size=batch, shuffle=False, collate_fn=pad_collate_tok)
    model = model.to(device)
    model.eval()

    out = []
    for Tok, Y, M, metas in ld:
        Tok, M = Tok.to(device), M.to(device)

        prob = torch.sigmoid(model(Tok, M)).cpu().numpy()  # shape [B, T]

        for b, (uid, days) in enumerate(metas):
            # days is the TRUE (unpadded) timeline for that user in this batch item
            for i, day in enumerate(days):
                out.append((uid, int(day), float(prob[b, i])))

    return pd.DataFrame(out, columns=["AgentID", "Day", "score"])



# ----------------------------
# 2) Splits (user-level, stratified)
# ----------------------------

def split_users_stratified_60_40(df, train_frac=0.60, seed=0):
    rng = np.random.default_rng(seed)
    users = df["AgentID"].unique()
    is_target = df.groupby("AgentID")["IsTarget"].first().reindex(users).values.astype(int)

    pos = users[is_target == 1]
    neg = users[is_target == 0]
    rng.shuffle(pos); rng.shuffle(neg)

    def _split(arr):
        n = len(arr)
        n_tr = int(train_frac * n)
        # guard: if possible, keep at least 1 in each split
        if n >= 2:
            n_tr = min(max(n_tr, 1), n - 1)
        tr = arr[:n_tr]
        te = arr[n_tr:]
        return tr, te

    tr_p, te_p = _split(pos)
    tr_n, te_n = _split(neg)

    tr = np.concatenate([tr_p, tr_n]); rng.shuffle(tr)
    te = np.concatenate([te_p, te_n]); rng.shuffle(te)
    return tr, te

def user_level_metrics(
    df: pd.DataFrame,
    score_col: str,
    theta: float,
    label_col: str = "y_killchain",
    attack_def: str = "killchain",      # "killchain" or "exfil"
    timing_scope: str = "label",        # kept for backward-compat; timing now uses ANY alert
    f1_mode: str = "period",            # "period" or "user_preexfil" or "user_ever"
):
    """
    Computes (exact, no approximations):
      - Pre-Exfil (%): P(first alert < first exfil) over attack users
      - Delta mean/median (days): E[t_alert - t_exfil] among detected attack users (can be negative)
      - MTTD (days): mean max(t_alert - t_exfil, 0) among detected attack users (post-exfil delay)
      - User FP (%): benign users with any alert (ever)
      - Mal FN (%) [legacy]: attack users with *no* alert on a positive-labeled period (original definition)
      - Attack FN Early (%): attack users with NO early warning (t_alert < t_exfil)
      - Attack FN Ever (%): attack users with NO alert ever
      - F1:
          * f1_mode="period": period-level F1 over label_col
          * f1_mode="user_preexfil": user-level EARLY-warning F1
              y_true(user)=1 if attack user else 0
              y_pred(user)=1 if (attack user alerted before exfil) OR (benign user ever alerted)
          * f1_mode="user_ever": user-level EVER-alert F1
              y_true(user)=1 if attack user else 0
              y_pred(user)=1 if user ever alerted (attack or benign)
    Notes:
      - Timing metrics use first alert = first pred==1 (ANY alert), regardless of timing_scope.
        (This matches "time to first alert relative to first exfil".)
      - The legacy Mal FN (%) remains label-gated to preserve  original definition.
    """
    d = df.copy()
    d["pred"] = (d[score_col].astype(float) >= float(theta)).astype(int)

    # ---------- PERIOD-LEVEL F1 (computed always; used only when f1_mode='period') ----------
    y_true_period = d[label_col].values.astype(int)
    y_pred_period = d["pred"].values.astype(int)
    f1_period = float(f1_score(y_true_period, y_pred_period, zero_division=0))

    # ---------- User sets ----------
    benign_users = d.loc[d["IsTarget"].astype(int) == 0, "AgentID"].unique()
    target = d[d["IsTarget"].astype(int) == 1].copy()

    # Any alert ever (benign)
    if len(benign_users) == 0:
        benign_any_alert = {}
        user_fp = 0.0
    else:
        benign_any_alert = (d[d["AgentID"].isin(benign_users)]
                            .groupby("AgentID")["pred"].max()
                            .to_dict())
        user_fp = 100.0 * float(np.mean(list(benign_any_alert.values()))) if benign_any_alert else 0.0

    # ---------- First EXFIL time per target user ----------
    t_ex = (target[target["Action"].astype(str) == "EXFIL"]
            .groupby("AgentID")["Day"].min())

    # ---------- Define ATTACK USERS ----------
    if attack_def == "exfil":
        attack_users = t_ex.index
    elif attack_def == "killchain":
        t_re = (target[target["Action"].astype(str) == "RECON"]
                .groupby("AgentID")["Day"].min())
        common = t_ex.index.intersection(t_re.index)
        attack_users = common[(t_re.loc[common].values <= t_ex.loc[common].values)]
    else:
        raise ValueError("attack_def must be 'killchain' or 'exfil'")

    # ---------- Legacy Mal FN (%) (original label-gated definition) ----------
    if len(attack_users) == 0:
        mal_fn = 0.0
    else:
        sub = target[target["AgentID"].isin(attack_users)]
        detected_on_label = sub.groupby("AgentID").apply(
            lambda g: int(((g["pred"] == 1) & (g[label_col] == 1)).any())
        )
        mal_fn = 100.0 * float(1.0 - detected_on_label.mean())

    # ---------- Timing + user-level predictions ----------
    pre_hits = []      # 1 if first alert < first exfil
    deltas = []        # t_alert - t_exfil (can be negative), only for detected users
    delays = []        # max(delta,0), only for detected users

    user_pred_early = {}   # for attack users: 1 if early warning
    user_pred_ever  = {}   # for attack users: 1 if any alert ever

    # Precompute attack-user any-alert quickly
    if len(attack_users) > 0:
        attack_any_alert = (target[target["AgentID"].isin(attack_users)]
                            .groupby("AgentID")["pred"].max()
                            .to_dict())
    else:
        attack_any_alert = {}

    for uid in attack_users:
        du = target[target["AgentID"] == uid]

        if uid not in t_ex.index:
            # Shouldn't happen for attack_users, but be safe
            pre_hits.append(0)
            user_pred_early[uid] = 0
            user_pred_ever[uid] = int(attack_any_alert.get(uid, 0))
            continue

        t_ex0 = int(t_ex.loc[uid])

        # IMPORTANT: timing uses ANY alert (first pred==1), not label-gated
        det_any = du.loc[(du["pred"] == 1), "Day"]
        t_det = int(det_any.min()) if len(det_any) else None

        ever = int(attack_any_alert.get(uid, 0))
        user_pred_ever[uid] = ever

        if t_det is None:
            pre_hits.append(0)
            user_pred_early[uid] = 0
        else:
            delta = int(t_det - t_ex0)
            deltas.append(delta)
            delays.append(max(delta, 0))
            hit = 1 if t_det < t_ex0 else 0
            pre_hits.append(hit)
            user_pred_early[uid] = hit

    pre_exfil = 100.0 * float(np.mean(pre_hits)) if len(pre_hits) else 0.0
    delta_mean = float(np.mean(deltas)) if len(deltas) else float("nan")
    delta_median = float(np.median(deltas)) if len(deltas) else float("nan")
    mttd = float(np.mean(delays)) if len(delays) else float("nan")

    # User-level FN rates aligned to the two objectives
    if len(attack_users) == 0:
        fn_early_pct = 0.0
        fn_ever_pct = 0.0
    else:
        fn_early_pct = 100.0 * float(1.0 - np.mean([user_pred_early.get(u, 0) for u in attack_users]))
        fn_ever_pct  = 100.0 * float(1.0 - np.mean([user_pred_ever.get(u, 0)  for u in attack_users]))

    # ---------- F1 selection ----------
    if f1_mode == "period":
        f1 = f1_period

    elif f1_mode == "user_preexfil":
        # Positives = attack_users. Pred positive:
        #   - attack: early warning hit
        #   - benign: any alert ever
        users_true_pos = set(attack_users.tolist()) if hasattr(attack_users, "tolist") else set(attack_users)
        users_true_neg = set(benign_users.tolist()) if hasattr(benign_users, "tolist") else set(benign_users)
        users_all = sorted(list(users_true_pos.union(users_true_neg)))

        y_u_true = []
        y_u_pred = []
        for uid in users_all:
            is_pos = 1 if uid in users_true_pos else 0
            if uid in users_true_pos:
                pred_pos = int(user_pred_early.get(uid, 0))
            else:
                pred_pos = int(benign_any_alert.get(uid, 0))
            y_u_true.append(is_pos)
            y_u_pred.append(pred_pos)

        f1 = float(f1_score(np.array(y_u_true), np.array(y_u_pred), zero_division=0))

    elif f1_mode == "user_ever":
        # Positives = attack_users. Pred positive = any alert ever (attack or benign).
        users_true_pos = set(attack_users.tolist()) if hasattr(attack_users, "tolist") else set(attack_users)
        users_true_neg = set(benign_users.tolist()) if hasattr(benign_users, "tolist") else set(benign_users)
        users_all = sorted(list(users_true_pos.union(users_true_neg)))

        y_u_true = []
        y_u_pred = []
        for uid in users_all:
            is_pos = 1 if uid in users_true_pos else 0
            if uid in users_true_pos:
                pred_pos = int(user_pred_ever.get(uid, 0))
            else:
                pred_pos = int(benign_any_alert.get(uid, 0))
            y_u_true.append(is_pos)
            y_u_pred.append(pred_pos)

        f1 = float(f1_score(np.array(y_u_true), np.array(y_u_pred), zero_division=0))

    else:
        raise ValueError("f1_mode must be 'period', 'user_preexfil', or 'user_ever'")

    return {
        "theta": float(theta),
        "Pre-Exfil (%)": pre_exfil,
        "Delta mean (days)": delta_mean,
        "Delta median (days)": delta_median,
        "MTTD (days)": mttd,
        "User FP (%)": user_fp,
        "Mal FN (%)": mal_fn,  # legacy (label-gated)
        "Attack FN Early (%)": fn_early_pct,
        "Attack FN Ever (%)": fn_ever_pct,
        "F1": f1,
    }

import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import Optional, Dict, List, Tuple

# --------- column autodetect helpers ----------

import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn

from sklearn.metrics import f1_score

def run_one(
    seed,
    N_users,
    ft,
    thresholds=(0.75, 0.85, 0.90),
    poison_rate=0.0,
    n_bins=8,
    epochs=20,
    batch=64,
    device=None,
):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    thresholds = ensure_iterable_thresholds(thresholds)
    rows = []

    # -------------------------
    # Behavioral sim (ground truth + split)
    # -------------------------
    set_seed(seed)
    globals()["NUM_USERS"] = N_users
    globals()["BEHAVIOR_MODE"] = "behavioral"

    df_btce = run_simulation(ft)
    df_btce = add_killchain_labels(df_btce)  # REQUIRED

    # Transformer labels MUST NOT be BTCE constructs.
    # Use sim/CERT-native labels derived from Action.
    df_btce["y_stage"] = (
        df_btce["Action"].astype(str).isin(["RECON", "EXFIL"]).astype(int)
    )
    df_btce["y_exfil_only"] = (
        (df_btce["Action"].astype(str) == "EXFIL").astype(int)
    )
    TRANS_LABEL = "y_stage"  # <-- use this everywhere for transformer training/scoring

    # Split users ONCE from deterministic ground truth
    tr_u, te_u = split_users_stratified_60_40(
        df_btce, train_frac=0.6, seed=seed
    )

    # Feature cols for transformer (must exclude s_exfil and any killchain constructs)
    DENY = {"s_exfil", "killchain"}  # belt + suspenders
    feat_cols = infer_transformer_feature_cols(df_btce, drop_cols=DENY)
    print("Transformer feature cols:", feat_cols)

    # -------------------------
    # Train Transformer UBS on TRAIN users only
    # (labels may remain y_killchain; features are sanitized)
    # -------------------------
    model = train_transformer_ubs(
        df_btce,
        feat_cols,
        tr_u,
        n_bins=n_bins,
        epochs=epochs,
        batch=batch,
        poison_rate=0.0,
        seed=seed,
        device=device,
        label_mode="user",
        user_label_col="IsTarget",
        prefix_train=True,
        min_prefix=3,  # strongly recommended
    )

    scores_te = score_transformer(
        df_btce,
        feat_cols,
        te_u,
        model,
        n_bins=n_bins,
        batch=batch,
        device=device,
        label_mode="user",
        user_label_col="IsTarget",
    )

    df_trans_te = (
        df_btce[df_btce["AgentID"].isin(te_u)]
        .merge(scores_te, on=["AgentID", "Day"], how="left")
    )
    df_trans_te["score"] = df_trans_te["score"].fillna(0.0)

    '''print(
        df_trans_te.groupby("Action")["score"].describe().to_string()
    )
    print()
    print(
        df_trans_te[df_trans_te["IsTarget"].astype(int) == 1]
        .groupby("Action")["score"]
        .describe()
        .to_string()
    )'''

    # BTCE eval on TEST users
    df_btce_te = df_btce[df_btce["AgentID"].isin(te_u)].copy()
    df_btce_te["P_Detect"] = df_btce_te["P_Detect"].astype(float)

    # -------------------------
    # Evaluate: BOTH user-level objectives for ALL methods
    # Behavioral also reports period-level F1 (allowed)
    # -------------------------
    for th in thresholds:
        # ---- BTCE Behavioral ----

        # Early-warning user-level
        m_be_early = user_level_metrics(
            df_btce_te,
            score_col="P_Detect",
            theta=th,
            label_col="y_killchain",
            attack_def="killchain",
            timing_scope="label",
            f1_mode="user_preexfil",
        )
        rows.append(("BTCE (Behavioral) - UserEarly", N_users, seed, m_be_early))

        # Ever-alert user-level
        m_be_ever = user_level_metrics(
            df_btce_te,
            score_col="P_Detect",
            theta=th,
            label_col="y_killchain",
            attack_def="killchain",
            timing_scope="label",
            f1_mode="user_ever",
        )
        rows.append(("BTCE (Behavioral) - UserEver", N_users, seed, m_be_ever))

        # Period-level
        m_be_period = user_level_metrics(
            df_btce_te,
            score_col="P_Detect",
            theta=th,
            label_col="y_killchain",
            attack_def="killchain",
            timing_scope="label",
            f1_mode="period",
        )
        rows.append(("BTCE (Behavioral) - Period", N_users, seed, m_be_period))

        # ---- Transformer-UBS (features sanitized, labels NOT BTCE constructs) ----
        df_trans_tmp = df_trans_te.rename(columns={"score": "TransformerScore"})

        m_tr_early = user_level_metrics(
            df_trans_tmp,
            score_col="TransformerScore",
            theta=th,
            label_col="y_stage",
            attack_def="exfil",
            timing_scope="any",
            f1_mode="user_preexfil",
        )
        rows.append(("Transformer-UBS - UserEarly", N_users, seed, m_tr_early))

        m_tr_ever = user_level_metrics(
            df_trans_tmp,
            score_col="TransformerScore",
            theta=th,
            label_col="y_stage",
            attack_def="exfil",
            timing_scope="any",
            f1_mode="user_ever",
        )
        rows.append(("Transformer-UBS - UserEver", N_users, seed, m_tr_ever))

    # -------------------------
    # Rational sim (same target set must match behavioral)
    # -------------------------
    set_seed(seed)
    globals()["NUM_USERS"] = N_users
    globals()["BEHAVIOR_MODE"] = "rational"

    df_rat = run_simulation(ft)
    df_rat = add_killchain_labels(df_rat)  # REQUIRED

    # Evaluate rational on same TEST users
    df_rat_te = df_rat[df_rat["AgentID"].isin(te_u)].copy()
    df_rat_te["P_Detect"] = df_rat_te["P_Detect"].astype(float)

    # Ensure target cohort identity matches across modes
    assert set(
        df_btce[df_btce.IsTarget.astype(int) == 1].AgentID.unique()
    ) == set(
        df_rat[df_rat.IsTarget.astype(int) == 1].AgentID.unique()
    ), "IsTarget mismatch across modes!"

    for th in thresholds:
        # Rational: do NOT report period-level F1.
        # Use EXFIL-based attack definition, and user-level objectives only.

        # Early-warning user-level
        m_rat_early = user_level_metrics(
            df_rat_te,
            score_col="P_Detect",
            theta=th,
            label_col="y_attack_anyexfil",
            attack_def="exfil",
            timing_scope="any",
            f1_mode="user_preexfil",
        )
        rows.append(("BTCE-Rational - UserEarly", N_users, seed, m_rat_early))

        # Ever-alert user-level
        m_rat_ever = user_level_metrics(
            df_rat_te,
            score_col="P_Detect",
            theta=th,
            label_col="y_attack_anyexfil",
            attack_def="exfil",
            timing_scope="any",
            f1_mode="user_ever",
        )
        rows.append(("BTCE-Rational - UserEver", N_users, seed, m_rat_ever))

    # -------------------------
    # Poisoned Transformer (optional)
    # -------------------------
    if poison_rate > 0:
        set_seed(seed)
        globals()["BEHAVIOR_MODE"] = "behavioral"

        model_p = train_transformer_ubs(
            df_btce,
            feat_cols,
            tr_u,
            n_bins=n_bins,
            epochs=epochs,
            batch=batch,
            poison_rate=poison_rate,
            seed=seed,
            device=device,
            label_mode="user",
            user_label_col="IsTarget",
            prefix_train=True,
            min_prefix=3,
        )

        scores_te_p = score_transformer(
            df_btce,
            feat_cols,
            te_u,
            model_p,
            n_bins=n_bins,
            batch=batch,
            device=device,
            label_mode="user",
            user_label_col="IsTarget",
        )

        df_p_te = (
            df_btce[df_btce["AgentID"].isin(te_u)]
            .merge(scores_te_p, on=["AgentID", "Day"], how="left")
        )
        df_p_te["score"] = df_p_te["score"].fillna(0.0)
        df_p_tmp = df_p_te.rename(columns={"score": "PoisonScore"})

        for th in thresholds:
            m_p_early = user_level_metrics(
                df_p_tmp,
                score_col="PoisonScore",
                theta=th,
                label_col="y_killchain",
                attack_def="killchain",
                timing_scope="label",
                f1_mode="user_preexfil",
            )
            rows.append(
                ("Transformer-UBS (poisoned) - UserEarly", N_users, seed, m_p_early)
            )

            m_p_ever = user_level_metrics(
                df_p_tmp,
                score_col="PoisonScore",
                theta=th,
                label_col="y_killchain",
                attack_def="killchain",
                timing_scope="label",
                f1_mode="user_ever",
            )
            rows.append(
                ("Transformer-UBS (poisoned) - UserEver", N_users, seed, m_p_ever)
            )

    return rows, df_btce, df_rat



# ----------------------------
# 4) Multi-run aggregation -> table DataFrame
# ----------------------------
def run_table(N=4000,
              thresholds=(0.75,0.85,0.90),
              n_runs=10,
              ft=3,
              poison_rate=0.20,
              n_bins=8,
              epochs=20,
              batch=64):
    all_rows = []
    device = "cuda" if torch.cuda.is_available() else "cpu"

    for seed in range(n_runs):
        rows, _, _ = run_one(
            seed=seed, N_users=N, ft=ft,
            thresholds=thresholds, poison_rate=poison_rate,
            n_bins=n_bins, epochs=epochs, batch=batch, device=device
        )
        all_rows.extend(rows)

    # ---- Flatten per-run rows ----
    flat = []
    for method, N_, seed, mdict in all_rows:
        flat.append({
            "Method": method,
            "N": N_,
            "seed": seed,
            **mdict
        })
    raw_runs = pd.DataFrame(flat)

    metric_cols = ["Pre-Exfil (%)", "Delta mean (days)", "Delta median (days)", "MTTD (days)", "User FP (%)", "Mal FN (%)", "F1"]

    # ---- Mean only (our current table) ----
    table_mean = (raw_runs.groupby(["Method", "N", "theta"], as_index=False)[metric_cols]
                          .mean())

    # ---- Mean + Std (numeric) ----
    agg_dict = {c: ["mean", "std"] for c in metric_cols}
    tmp = (raw_runs.groupby(["Method", "N", "theta"], as_index=False)
                  .agg(agg_dict))

    # flatten MultiIndex columns like ('F1','mean') -> 'F1_mean'
    tmp.columns = [
        "_".join([c for c in col if c != ""]) if isinstance(col, tuple) else col
        for col in tmp.columns
    ]
    # rename group keys back (they become 'Method_', etc if we’re not careful)
    tmp = tmp.rename(columns={
        "Method_": "Method",
        "N_": "N",
        "theta_": "theta",
    })

    # ---- Mean±Std formatted strings (nice for LaTeX) ----
    def fmt_mean_std(m, s, decimals=2):
        if pd.isna(m): return ""
        if pd.isna(s): s = 0.0
        return f"{m:.{decimals}f} $\\pm$ {s:.{decimals}f}"

    table_mean_std = tmp.copy()
    # choose decimals per metric
    dec = {
        "Pre-Exfil (%)": 2,
        "Delta mean (days)": 2,
        "Delta median (days)": 2,
        "MTTD (days)": 2,
        "User FP (%)": 2,
        "Mal FN (%)": 2,
        "F1": 3,
    }

    for c in metric_cols:
        mcol, scol = f"{c}_mean", f"{c}_std"
        table_mean_std[c] = [
            fmt_mean_std(m, s, decimals=dec[c])
            for m, s in zip(table_mean_std[mcol], table_mean_std[scol])
        ]

    # keep only the formatted columns + keys, but we still have numeric *_mean/*_std in `tmp`
    table_mean_std = table_mean_std[["Method", "N", "theta"] + metric_cols]

    return raw_runs, table_mean, tmp, table_mean_std

