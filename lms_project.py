"""
ECE 5523 Project -- LMS Adaptive Filter on AWGN, Phase-Drift and Rayleigh Channels
Author : Dayvon "Mac" McCarrell (LMS section)
Python : 3.9 or newer.  Packages: numpy, scipy, matplotlib
         Install once with:   pip install numpy scipy matplotlib
Run    : python lms_project.py
         Takes about 40 seconds. Saves four figures in a folder called "figures"
         and prints every number used in the report.

WHAT THIS PROGRAM DOES, IN PLAIN LANGUAGE
  1. Makes a random test signal (a smooth, "colored" random waveform).
  2. Sends it through three kinds of channel damage: added noise (AWGN),
     a slowly wandering phase (phase drift), and fading (Rayleigh).
  3. Uses an LMS adaptive filter to recover the clean signal from the damaged one.
     The filter "learns" its own settings by nudging them a little after every
     sample to reduce the error it just made.
  4. Measures how close it gets (mean-square error, MSE) and how many bits it gets
     wrong (bit error rate, BER), and compares both against textbook theory.

HOW THE FILE IS ORGANIZED (read top to bottom; each step uses only earlier steps)
  STEP 0  Settings you may change
  STEP 1  The random test signal
  STEP 2  The three channels
  STEP 3  The filter's input window (the last M received samples)
  STEP 4  The LMS / NLMS filter itself
  STEP 5  Theory formulas used to check the simulation
  STEP 6  Experiment A: learning curves and misadjustment       -> figure 1
  STEP 7  Experiment B: error versus signal-to-noise ratio       -> figure 2
  STEP 8  Experiment C: error versus step size (tracking)        -> figure 3
  STEP 9  Experiment D: bit error rate, plus a stability check    -> figure 4
  STEP 10 Main program: runs everything and prints the summary

KEY EQUATIONS (same symbols as the report)
  filter output     y(n) = w^H u(n)        (weighted sum of the last M inputs)
  error             e(n) = d(n) - y(n)      (wanted signal minus what we produced)
  LMS update        w(n+1) = w(n) + mu * u(n) * conj(e(n))

A NOTE ON SPEED
  Every array has shape (trials, samples). One loop walks through time, and each
  line inside it updates all Monte Carlo trials at once, which keeps Python fast.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")              # save plots to files instead of opening windows
import matplotlib.pyplot as plt
from scipy.special import j0, erfc  # j0 = Bessel function (fading), erfc = for BER
from scipy.stats import chi2        # chi-square distribution (whiteness test)


# =====================================================================================
# STEP 0 -- SETTINGS
# Change these numbers to explore. Everything else in the program follows from them.
# =====================================================================================
SEED        = 5523      # fixed random seed, so every run gives identical results
M_TAPS      = 8         # how many past samples the filter looks at (filter length)
RHO         = 0.95      # how "smooth" the test signal is: 0 = pure noise, near 1 = slow
PHASE_STD   = 0.01      # how fast the phase wanders, in radians per sample
FD_T        = 0.002     # how fast the fading changes (Doppler x sample time)
N_SAMPLES   = 6000      # length of each simulated record
N_TRIALS    = 60        # number of independent repeats averaged together
SNR_DB_LIST = [0, 5, 10, 15, 20, 25, 30]                # signal-to-noise ratios tested
MU_GRID     = [0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2]  # LMS step sizes (x tr(R))
NLMS_GRID   = [0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.8]      # NLMS step sizes
FIG_DIR     = "figures"

rng = np.random.default_rng(SEED)   # the single random-number generator for the program


def cgauss(shape, var=1.0):
    """Complex Gaussian noise with total power `var`.
    Real and imaginary parts are independent, each carrying half the power."""
    return np.sqrt(var / 2) * (rng.standard_normal(shape)
                               + 1j * rng.standard_normal(shape))


# =====================================================================================
# STEP 1 -- THE RANDOM TEST SIGNAL
# =====================================================================================
def ar1_signal(trials, n, rho=RHO):
    """Smooth random signal (first-order autoregressive, "AR(1)").
    Each new sample = rho x previous sample + a little fresh randomness:
        s(n) = rho*s(n-1) + sqrt(1 - rho^2) * v(n)
    Neighboring samples are similar (correlation rho), so a filter can use the
    past to clean up the present. Average power is 1 from the very first sample."""
    s = np.empty((trials, n), dtype=complex)
    v = cgauss((trials, n))                  # the fresh randomness for every sample
    s[:, 0] = v[:, 0]                        # start already at full power
    g = np.sqrt(1 - rho ** 2)                # keeps the power at exactly 1
    for k in range(1, n):
        s[:, k] = rho * s[:, k - 1] + g * v[:, k]
    return s


def bpsk_symbols(trials, n):
    """Random data bits sent as +1 or -1 (BPSK), used for the bit-error-rate test."""
    return 2.0 * rng.integers(0, 2, size=(trials, n)) - 1.0


# =====================================================================================
# STEP 2 -- THE THREE CHANNELS
# Each function takes the clean signal and returns what the receiver sees.
# Share these with the Wiener and Kalman code so every filter faces the same damage.
# =====================================================================================
def channel_awgn(s, noise_var):
    """Additive white Gaussian noise: received = signal + random noise.
        r(n) = s(n) + v(n)"""
    return s + cgauss(s.shape, noise_var)


def channel_phase_drift(s, noise_var, phase_std=PHASE_STD):
    """Phase drift: the signal is rotated by an angle that wanders randomly,
    like a receiver clock that slowly slips, and then noise is added.
        r(n) = s(n) * exp(j*theta(n)) + v(n),   theta(n) = theta(n-1) + small step
    The angle is a random walk (Wiener process). Its spread grows over time,
    so this channel is NOT stationary."""
    theta0 = rng.uniform(-np.pi, np.pi, size=(s.shape[0], 1))       # random start angle
    steps = phase_std * rng.standard_normal(s.shape)                  # small random steps
    theta = theta0 + np.cumsum(steps, axis=1)                         # running total
    return s * np.exp(1j * theta) + cgauss(s.shape, noise_var)


def rayleigh_gain(trials, n, fd_t=FD_T):
    """Fading gain h(n): a complex number whose size and angle change slowly,
    like a mobile radio moving through reflections. Built the same way as the
    test signal, with smoothness a = J0(2*pi*fD*T) taken from Clarke's model
    (method of Baddour & Beaulieu, 2005)."""
    a = j0(2 * np.pi * fd_t)
    h = np.empty((trials, n), dtype=complex)
    q = cgauss((trials, n))
    h[:, 0] = q[:, 0]
    g = np.sqrt(1 - a ** 2)
    for k in range(1, n):
        h[:, k] = a * h[:, k - 1] + g * q[:, k]
    return h


def channel_rayleigh(s, noise_var, fd_t=FD_T):
    """Rayleigh fading: the signal is multiplied by the fading gain, then noise is
    added.  r(n) = h(n) * s(n) + v(n).  When |h| is small (a "deep fade") the
    signal almost disappears under the noise."""
    h = rayleigh_gain(*s.shape, fd_t=fd_t)
    return h * s + cgauss(s.shape, noise_var)


CHANNELS = {"AWGN": channel_awgn,
            "Phase drift": channel_phase_drift,
            "Rayleigh": channel_rayleigh}


# =====================================================================================
# STEP 3 -- THE FILTER'S INPUT WINDOW
# At time n the filter looks at the newest M received samples:
#     u(n) = [x(n), x(n-1), ..., x(n-M+1)]
# This step builds that window for every time n at once.
# =====================================================================================
def build_regressors(x, m):
    """Input: x with shape (trials, N).  Output: U with shape (trials, N, M),
    where U[:, n, k] is the sample from k steps ago (zero before the record starts)."""
    pad = np.zeros((x.shape[0], m - 1), dtype=x.dtype)   # zeros before time 0
    xp = np.concatenate([pad, x], axis=1)
    win = np.lib.stride_tricks.sliding_window_view(xp, m, axis=1)   # oldest .. newest
    return win[:, :, ::-1]                                          # newest first


# =====================================================================================
# STEP 4 -- THE LMS / NLMS ADAPTIVE FILTER
# This is the whole algorithm. Every sample it does five things:
#   (a) start with all weights at zero (only once, before the loop)
#   (b) make an output: a weighted sum of the samples in the window
#   (c) decide what the output SHOULD have been (the training signal)
#   (d) measure the error
#   (e) nudge every weight a little in the direction that shrinks the error
# The step size mu sets how big each nudge is: big = learns fast but jittery,
# small = smooth but slow. NLMS divides the nudge by the input power so the
# nudge stays the right size even when the signal gets very loud or very quiet.
# =====================================================================================
def lms(U, d, mu, normalized=False, eps=1e-6, pilot_mask=None, decide=None):
    """Complex LMS (Widrow, McCool & Ball, 1975) or NLMS (Nagumo & Noda, 1967).

    U          : input windows from build_regressors, shape (trials, N, M)
    d          : what the output should be (the clean signal or known symbols)
    mu         : step size. For NLMS use a value between 0 and 2.
    normalized : False = LMS, True = NLMS
    eps        : tiny number that stops NLMS dividing by zero
    pilot_mask : optional True/False list over time. True = the correct answer is
                 known at that sample (a "pilot"). When it is False the filter either
                 trusts its own decision (if `decide` is given) or skips learning.
    Returns the filter output y and the final weights w."""
    trials, n_samp, m = U.shape
    w = np.zeros((trials, m), dtype=complex)         # (a) all weights start at zero
    y = np.empty((trials, n_samp), dtype=complex)
    for n in range(n_samp):
        u = U[:, n, :]                               # the current input window
        y[:, n] = np.sum(np.conj(w) * u, axis=1)     # (b) output y = w^H u
        if pilot_mask is None or pilot_mask[n]:
            dn = d[:, n]                             # (c) known correct answer
        elif decide is None:
            continue                                 #     unknown: do not learn here
        else:
            dn = decide(y[:, n])                     #     unknown: trust own decision
        e = dn - y[:, n]                             # (d) error = wanted - produced
        if normalized:                               # NLMS: scale by input power
            step = mu / (eps + np.sum(np.abs(u) ** 2, axis=1))
        else:                                        # LMS: fixed step
            step = mu
        w += (step * np.conj(e))[:, None] * u        # (e) w <- w + mu * u * conj(e)
    return y, w


# =====================================================================================
# STEP 5 -- THEORY FORMULAS
# The simulation is only trustworthy if it agrees with theory where theory applies.
# =====================================================================================
def awgn_R(m, noise_var, rho=RHO):
    """Input correlation matrix R for signal + AWGN: how strongly each input sample
    is related to each other one. R[i,j] = rho^|i-j|, plus the noise power on
    the diagonal."""
    k = np.arange(m)
    return rho ** np.abs(k[:, None] - k[None, :]) + noise_var * np.eye(m)


def wiener_awgn(m, noise_var, rho=RHO):
    """Best possible fixed M-tap filter for the AWGN channel (the Wiener filter).
    Solves R w_o = p, where p[k] = rho^k measures how each input relates to the
    wanted sample. J_min is the smallest MSE any fixed filter of this length can get.
    Returns (w_o, J_min, trace of R)."""
    R = awgn_R(m, noise_var, rho)
    p = rho ** np.arange(m)
    w_o = np.linalg.solve(R, p)
    j_min = 1.0 - p @ w_o
    return w_o, j_min, np.trace(R)


def misadjustment_theory(R, mu):
    """Misadjustment = how much extra error LMS has above J_min, because its
    weights never sit perfectly still. Independence-theory formula:
        S = sum over eigenvalues lam of  mu*lam / (2 - mu*lam),   M = S / (1 - S)
    For small mu this is about mu * trace(R) / 2."""
    lam = np.linalg.eigvalsh(R)
    S = np.sum(mu * lam / (2 - mu * lam))
    return S / (1 - S)


def lms_steady_state_theory(j_min, trR, mu, R=None):
    """Predicted final LMS error: J(inf) = J_min * (1 + misadjustment).
    Uses the exact formula when R is given, otherwise the small-step shortcut."""
    mis = misadjustment_theory(R, mu) if R is not None else mu * trR / 2
    return j_min * (1 + mis)


def lms_tracking_theory(j_min, trR, trQ, mu):
    """Predicted LMS error when the best weights keep moving (a random walk with
    step power trQ). Two penalties pull in opposite directions:
        jitter penalty  grows with mu   ->  0.5 * mu * J_min * trace(R)
        lag penalty     shrinks with mu ->  0.5 * trQ / mu
    The best step size balances the two."""
    return j_min + 0.5 * (mu * j_min * trR + trQ / mu)


def qfunc(x):
    """Gaussian tail probability Q(x): chance a standard normal value exceeds x."""
    return 0.5 * erfc(x / np.sqrt(2))


def ber_bpsk_awgn(ebn0):
    """Theoretical BPSK bit error rate on AWGN with perfect phase knowledge."""
    return qfunc(np.sqrt(2 * ebn0))


def ber_bpsk_rayleigh(ebn0):
    """Theoretical BPSK bit error rate on Rayleigh fading with perfect channel knowledge."""
    return 0.5 * (1 - np.sqrt(ebn0 / (1 + ebn0)))


def ljung_box(e, lags=20):
    """Ljung-Box test (1978): is a sequence white (no memory)? Adds up its squared
    autocorrelations over `lags` steps. A small p-value means NOT white.
    Returns (Q statistic, p-value)."""
    e = e - e.mean()
    n = len(e)
    r0 = np.vdot(e, e).real
    r = np.array([np.vdot(e[:-k], e[k:]).real / r0 for k in range(1, lags + 1)])
    q = n * (n + 2) * np.sum(r ** 2 / (n - np.arange(1, lags + 1)))
    return q, 1 - chi2.cdf(q, lags)


def steady_mse(y, s, frac=0.5):
    """Average squared error |s - y|^2 over the last half of the record (after the
    filter has settled), averaged across all trials."""
    start = int(len(s[0]) * (1 - frac))
    return np.mean(np.abs(s[:, start:] - y[:, start:]) ** 2)


def db(x):
    """Convert a power ratio to decibels."""
    return 10 * np.log10(x)


# =====================================================================================
# STEP 6 -- EXPERIMENT A: LEARNING CURVES AND MISADJUSTMENT (figure 1)
# Watch the error fall as the filter learns, for three step sizes, and check the
# final level against theory.
# =====================================================================================
def experiment_learning_curves(snr_db=10, n=8000, trials=200,
                               norm_steps=(0.02, 0.1, 0.25)):
    noise_var = 10 ** (-snr_db / 10)                 # SNR in dB -> noise power
    w_o, j_min, trR = wiener_awgn(M_TAPS, noise_var)
    R = awgn_R(M_TAPS, noise_var)
    s = ar1_signal(trials, n)
    U = build_regressors(channel_awgn(s, noise_var), M_TAPS)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    rows = []
    for i, ns in enumerate(norm_steps):
        mu = ns / trR                                # step size scaled by input power
        y, _ = lms(U, s, mu)
        curve = np.mean(np.abs(s - y) ** 2, axis=0)  # average error at each time step
        smooth = np.convolve(curve, np.ones(25) / 25, mode="valid")   # light smoothing
        shade = 0.35 + 0.3 * i
        ax.plot(np.arange(len(smooth)) + 12, db(smooth),
                color=(0.1, 0.3, 0.7, shade), lw=1.3, label=f"LMS, mu*tr(R) = {ns}")
        j_th = lms_steady_state_theory(j_min, trR, mu, R)
        ax.axhline(db(j_th), color=(0.1, 0.3, 0.7, shade), ls="--", lw=0.9)
        rows.append((ns, mu, db(steady_mse(y, s)), db(j_th)))
    ax.axhline(db(j_min), color="k", lw=1.0, label=f"Wiener J_min = {db(j_min):.2f} dB")
    ax.set_xlabel("Sample n")
    ax.set_ylabel("Ensemble MSE (dB)")
    ax.set_title(f"LMS learning curves, AWGN, SNR = {snr_db} dB, M = {M_TAPS}\n"
                 f"(dashed = independence theory J_min(1 + M))")
    ax.set_xlim(0, n)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig1_learning_curves_awgn.png"), dpi=150)
    plt.close(fig)
    return rows, j_min


def experiment_misadjustment(snr_db=10, n=12000, trials=200, rhos=(0.0, 0.5, 0.95),
                             norm_steps=(0.05, 0.1, 0.25)):
    """Compare simulated and theoretical misadjustment for signals from pure noise
    (rho = 0) to very smooth (rho = 0.95). The theory assumes each input window is
    unrelated to the last one; the smooth signal shows what happens when that is
    false."""
    noise_var = 10 ** (-snr_db / 10)
    table = []
    for rho in rhos:
        _, j_min, trR = wiener_awgn(M_TAPS, noise_var, rho)
        R = awgn_R(M_TAPS, noise_var, rho)
        s = ar1_signal(trials, n, rho)
        U = build_regressors(channel_awgn(s, noise_var), M_TAPS)
        for ns in norm_steps:
            y, _ = lms(U, s, ns / trR)
            m_sim = steady_mse(y, s) / j_min - 1     # extra error, as a fraction
            table.append((rho, ns, 100 * m_sim,
                          100 * misadjustment_theory(R, ns / trR), 100 * ns / 2))
    return table


# =====================================================================================
# STEP 7 -- EXPERIMENT B: ERROR VERSUS SNR ON EVERY CHANNEL (figure 2)
# For a fair comparison each filter gets its best step size at every point.
# =====================================================================================
def best_over_mu(U, s, trR, normalized):
    """Try every step size on the grid and keep the one with the lowest error."""
    best = (np.inf, None)
    for ns in (NLMS_GRID if normalized else MU_GRID):
        mu = ns if normalized else ns / trR
        y, _ = lms(U, s, mu, normalized=normalized)
        mse = steady_mse(y, s)
        if mse < best[0]:
            best = (mse, ns)
    return best


def experiment_mse_vs_snr():
    results = {name: {"raw": [], "lms": [], "nlms": [], "lms_mu": [], "nlms_mu": [],
                      "wiener": []} for name in CHANNELS}
    for snr_db in SNR_DB_LIST:
        noise_var = 10 ** (-snr_db / 10)
        _, j_min, trR = wiener_awgn(M_TAPS, noise_var)
        for name, chan in CHANNELS.items():
            s = ar1_signal(N_TRIALS, N_SAMPLES)
            x = chan(s, noise_var)
            U = build_regressors(x, M_TAPS)
            r = results[name]
            r["raw"].append(steady_mse(x, s))        # baseline: no filter at all
            mse, ns = best_over_mu(U, s, trR, normalized=False)
            r["lms"].append(mse); r["lms_mu"].append(ns)
            mse, ns = best_over_mu(U, s, trR, normalized=True)
            r["nlms"].append(mse); r["nlms_mu"].append(ns)
            r["wiener"].append(j_min)                # best fixed filter on AWGN
        print(f"  SNR {snr_db:2d} dB done")
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), sharey=True)
    for ax, name in zip(axes, CHANNELS):
        r = results[name]
        ax.plot(SNR_DB_LIST, db(r["raw"]), "o-", color="0.6", label="No filter (r(n) as is)")
        ax.plot(SNR_DB_LIST, db(r["lms"]), "s-", color="tab:blue", label="LMS (tuned mu)")
        ax.plot(SNR_DB_LIST, db(r["nlms"]), "^--", color="tab:blue", alpha=0.6,
                label="NLMS (tuned mu)")
        ax.plot(SNR_DB_LIST, db(r["wiener"]), "k:", label="FIR Wiener J_min (AWGN bound)")
        ax.set_title(name)
        ax.set_xlabel("SNR (dB)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("Steady-state MSE (dB)")
    axes[0].legend(fontsize=8)
    fig.suptitle(f"LMS signal estimation, M = {M_TAPS} taps, AR(1) signal rho = {RHO}")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig2_mse_vs_snr.png"), dpi=150)
    plt.close(fig)
    return results


# =====================================================================================
# STEP 8 -- EXPERIMENT C: ERROR VERSUS STEP SIZE (figure 3)
# On a moving channel, too small a step lags behind and too large a step jitters.
# This sweep finds the sweet spot and compares it with the tracking formula.
# =====================================================================================
def experiment_mse_vs_mu(snr_db=20):
    noise_var = 10 ** (-snr_db / 10)
    w_o, j_min, trR = wiener_awgn(M_TAPS, noise_var)
    steps = np.logspace(-3, np.log10(0.3), 13)       # 13 step sizes, 0.001 to 0.3
    # any error above 0 dB means the filter blew up; leave those points blank
    ok = lambda v: np.where(np.asarray(v) < 1.0, np.asarray(v), np.nan)
    curves = {}
    for name, chan in CHANNELS.items():
        s = ar1_signal(N_TRIALS, N_SAMPLES)
        U = build_regressors(chan(s, noise_var), M_TAPS)
        curves[name] = [steady_mse(lms(U, s, ns / trR)[0], s) for ns in steps]
    mus = steps / trR
    # Theory for phase drift: the best weights must rotate with the phase, so each
    # sample they move by about (phase step) x (weight size). That gives the
    # "lag" power trQ = PHASE_STD^2 * ||w_awgn||^2 used in the tracking formula.
    trQ = PHASE_STD ** 2 * np.sum(np.abs(w_o) ** 2)
    th_track = lms_tracking_theory(j_min, trR, trQ, mus)
    R = awgn_R(M_TAPS, noise_var)
    th_stat = np.array([lms_steady_state_theory(j_min, trR, m, R) for m in mus])
    mu_opt = np.sqrt(trQ / (j_min * trR))           # step where jitter = lag
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ax.semilogx(steps, db(ok(curves["AWGN"])), "o-", color="0.55", label="AWGN (sim)")
    ax.semilogx(steps, db(th_stat), ":", color="0.55", label="AWGN independence theory")
    ax.semilogx(steps, db(ok(curves["Phase drift"])), "s-", color="tab:blue",
                label="Phase drift (sim)")
    ax.semilogx(steps, db(th_track), "--", color="tab:blue",
                label="Phase-drift tracking theory")
    ax.semilogx(steps, db(ok(curves["Rayleigh"])), "^-", color="tab:blue", alpha=0.45,
                label="Rayleigh (sim)")
    ax.set_ylim(-22, -6)
    ax.axvline(mu_opt * trR, color="tab:blue", lw=0.8, alpha=0.6)
    ax.text(mu_opt * trR * 0.97, -6.8, "theory mu_opt", color="tab:blue", fontsize=8,
            ha="right")
    ax.text(0.0011, -21.3, "Missing points = some trials diverged (MSE > 0 dB)",
            fontsize=7.5, color="0.4")
    ax.set_xlabel("Normalised step size  mu * tr(R)")
    ax.set_ylabel("Steady-state MSE (dB)")
    ax.set_title(f"Gradient noise vs lag: MSE against step size, SNR = {snr_db} dB\n"
                 f"(MSE over samples {N_SAMPLES // 2}-{N_SAMPLES}; "
                 f"small steps have not fully converged)", fontsize=10)
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig3_mse_vs_mu.png"), dpi=150)
    plt.close(fig)
    best = {name: steps[int(np.argmin(c))] for name, c in curves.items()}
    return best, mu_opt * trR, steps, curves


# =====================================================================================
# STEP 9 -- EXPERIMENT D: BIT ERROR RATE (figure 4) AND A STABILITY CHECK
# =====================================================================================
def experiment_ber(ebn0_db_list=range(0, 31, 3), n=20000, trials=40, pilot_every=5,
                   mu_tilde=0.5):
    """Send +/-1 data bits through each channel. A ONE-weight NLMS filter (a flat
    channel only needs one gain and one phase correction) learns from known
    "pilot" bits, one in every `pilot_every`, and holds its setting between them.
    Bit errors are counted on the data bits only, after the first 10% of the burst."""
    pilot_mask = (np.arange(n) % pilot_every) == 0   # True on every 5th bit
    ber = {name: [] for name in CHANNELS}
    ber_raw_phase = []
    ebn0_db_list = list(ebn0_db_list)
    data = ~pilot_mask                               # the bits we actually score
    data[: n // 10] = False                          # skip the start-up period
    for ebn0_db in ebn0_db_list:
        n0 = 10 ** (-ebn0_db / 10)                   # noise power for this Eb/N0
        for name, chan in CHANNELS.items():
            b = bpsk_symbols(trials, n)
            x = chan(b.astype(complex), n0)
            y, _ = lms(build_regressors(x, 1), b.astype(complex), mu_tilde,
                       normalized=True, eps=n0, pilot_mask=pilot_mask, decide=None)
            # a bit is wrong when the sign of the output disagrees with the sent bit
            ber[name].append(np.mean(np.sign(y[:, data].real) != b[:, data]))
            if name == "Phase drift":                # also score "no correction"
                ber_raw_phase.append(np.mean(np.sign(x[:, data].real) != b[:, data]))
    e = 10 ** (np.array(ebn0_db_list) / 10)
    clip = lambda v: np.where(np.asarray(v) > 0, v, np.nan)   # hide zero-error points
    fig, ax = plt.subplots(figsize=(7.5, 4.8))
    ax.semilogy(ebn0_db_list, ber_bpsk_awgn(e), "k-", lw=1,
                label="Theory: coherent BPSK, AWGN")
    ax.semilogy(ebn0_db_list, ber_bpsk_rayleigh(e), "k--", lw=1,
                label="Theory: coherent BPSK, Rayleigh")
    ax.semilogy(ebn0_db_list, clip(ber["AWGN"]), "o", ms=9, mfc="none", color="0.3",
                label="NLMS, AWGN")
    ax.semilogy(ebn0_db_list, clip(ber["Phase drift"]), "s-", color="tab:blue",
                label="NLMS, phase drift")
    ax.semilogy(ebn0_db_list, clip(ber_raw_phase), "x:", color="0.6",
                label="No correction, phase drift")
    ax.semilogy(ebn0_db_list, clip(ber["Rayleigh"]), "^-", color="tab:blue", alpha=0.5,
                label="NLMS, Rayleigh")
    ax.set_ylim(1e-6, 1)
    ax.text(0.3, 1.6e-6, f"Points with 0 errors in {trials * data.sum():,} bits "
            f"are not drawn", fontsize=7.5, color="0.4")
    ax.set_xlabel("Eb/N0 (dB)")
    ax.set_ylabel("Bit error rate")
    ax.set_title(f"BER with pilot-aided one-tap NLMS (1 pilot every {pilot_every} symbols)")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "fig4_ber.png"), dpi=150)
    plt.close(fig)
    return ebn0_db_list, ber, ber_raw_phase


def experiment_stability(snr_db=10, n=4000, trials=300,
                         steps=(0.1, 0.2, 0.3, 0.4, 0.6, 1.0)):
    """How often does the filter blow up as the step size grows? Counts the share of
    trials whose error ends above 0 dB. Textbooks allow steps up to 2 (x tr(R));
    Feuer & Weinstein (1985) showed the real limit for Gaussian data is lower."""
    noise_var = 10 ** (-snr_db / 10)
    _, _, trR = wiener_awgn(M_TAPS, noise_var)
    s = ar1_signal(trials, n)
    U = build_regressors(channel_awgn(s, noise_var), M_TAPS)
    out = []
    with np.errstate(all="ignore"):                  # blown-up trials overflow; that's ok
        for ns in steps:
            y, _ = lms(U, s, ns / trR)
            per_trial = np.mean(np.abs(s[:, n // 2:] - y[:, n // 2:]) ** 2, axis=1)
            yn, _ = lms(U, s, ns, normalized=True)   # NLMS with the same nominal step
            per_trial_n = np.mean(np.abs(s[:, n // 2:] - yn[:, n // 2:]) ** 2, axis=1)
            bad = lambda m: np.mean(~np.isfinite(m) | (m > 1.0))
            out.append((ns, bad(per_trial), bad(per_trial_n)))
    return out


# =====================================================================================
# STEP 10 -- MAIN PROGRAM: run every experiment and print the results
# =====================================================================================
def main():
    np.seterr(all="ignore")       # blown-up trials at big steps are expected and handled
    os.makedirs(FIG_DIR, exist_ok=True)

    print("Experiment A: learning curves (AWGN, 10 dB)")
    rows, j_min = experiment_learning_curves()
    print(f"  Wiener J_min = {db(j_min):.2f} dB")
    for ns, mu, sim, th in rows:
        print(f"  mu*tr(R) = {ns:<5}  mu = {mu:.5f}   sim J(inf) = {sim:6.2f} dB"
              f"   theory = {th:6.2f} dB")

    print("Misadjustment check (AWGN, 10 dB): simulated vs independence theory, in %")
    print(f"  {'rho':>5}{'mu*trR':>8}{'sim M':>9}{'exact th':>10}{'small-mu th':>13}")
    for rho, ns, ms, me, ma in experiment_misadjustment():
        print(f"  {rho:>5}{ns:>8}{ms:>9.2f}{me:>10.2f}{ma:>13.2f}")

    print("Experiment B: MSE vs SNR (tuned step sizes)")
    res = experiment_mse_vs_snr()

    print("Experiment C: MSE vs step size (20 dB)")
    best, mu_opt_th, steps, curves = experiment_mse_vs_mu()
    print(f"  Phase drift: theory mu_opt*tr(R) = {mu_opt_th:.3f}, "
          f"simulated best = {best['Phase drift']:.3f}")
    print(f"  Rayleigh   : simulated best mu*tr(R) = {best['Rayleigh']:.3f}")
    print(f"  AWGN       : simulated best mu*tr(R) = {best['AWGN']:.3f} "
          f"(set by convergence speed in a finite record)")

    print("Experiment D: BER (pilot-aided one-tap NLMS)")
    ebn0, ber, ber_raw = experiment_ber()

    print("Stability check: fraction of diverged trials (AWGN, 10 dB, M = 8)")
    for ns, f_lms, f_nlms in experiment_stability():
        print(f"  step {ns:<4}  LMS diverged {100*f_lms:5.1f}%"
              f"   NLMS diverged {100*f_nlms:5.1f}%")

    # Is the LMS error white? (ties to the project question about white vs colored noise)
    noise_var = 10 ** (-20 / 10)
    _, _, trR = wiener_awgn(M_TAPS, noise_var)
    s = ar1_signal(1, 20000)
    y, _ = lms(build_regressors(channel_awgn(s, noise_var), M_TAPS), s, 0.05 / trR)
    q_lb, p_lb = ljung_box((s - y)[0, 10000:])

    print("\nSUMMARY at SNR = 20 dB (steady-state MSE, dB)")
    i20 = SNR_DB_LIST.index(20)
    print(f"{'Channel':<13}{'No filter':>10}{'LMS':>9}{'NLMS':>9}{'LMS mu*trR':>12}"
          f"{'Wiener Jmin':>13}")
    for name in CHANNELS:
        r = res[name]
        print(f"{name:<13}{db(r['raw'][i20]):>10.2f}{db(r['lms'][i20]):>9.2f}"
              f"{db(r['nlms'][i20]):>9.2f}{r['lms_mu'][i20]:>12}"
              f"{db(r['wiener'][i20]):>13.2f}")
    print("\nBER at Eb/N0 = 9, 18, 27 dB:")
    for name in CHANNELS:
        print(f"  {name:<12} "
              + "   ".join(f"{ber[name][ebn0.index(v)]:.2e}" for v in (9, 18, 27)))
    print(f"  Phase drift, no correction: {ber_raw[ebn0.index(9)]:.2e}")
    print("  Theory AWGN     "
          + "   ".join(f"{ber_bpsk_awgn(10**(v/10)):.2e}" for v in (9, 18, 27)))
    print("  Theory Rayleigh "
          + "   ".join(f"{ber_bpsk_rayleigh(10**(v/10)):.2e}" for v in (9, 18, 27)))
    print(f"\nLjung-Box on LMS error (AWGN, 20 dB, 20 lags): Q = {q_lb:.1f}, p = {p_lb:.3g}")
    np.savez(os.path.join(FIG_DIR, "results.npz"), snr_db=SNR_DB_LIST,
             **{f"{k}_{m}".replace(" ", "_"): np.array(v)
                for k, r in res.items() for m, v in r.items()})


if __name__ == "__main__":
    main()
