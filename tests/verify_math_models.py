"""
NTRO SIH26162 Architecture Mathematical & Physical Verification Suite
Challenger 1 Verification Engine
"""
import math
import numpy as np
from scipy import integrate, optimize

def verify_thermal_physics():
    print("==================================================")
    print("1. THERMAL PHYSICS VERIFICATION")
    print("==================================================")
    # Physical Constants (CODATA 2018 / SI 2019 redefinition)
    h = 6.62607015e-34       # J*s
    c = 2.99792458e8         # m/s
    k_B = 1.380649e-23       # J/K
    sigma_exact = (2 * math.pi**5 * k_B**4) / (15 * c**2 * h**3)
    c1 = 2 * math.pi * h * c**2 # W*m^2
    c2 = (h * c) / k_B       # m*K
    c2_um_K = c2 * 1e6       # um*K

    print(f"c1 = {c1:.6e} W*m^2 (Text: 3.74177e-16)")
    print(f"c2 = {c2:.6e} m*K = {c2_um_K:.4f} um*K (Text: 14387.8 um*K)")
    print(f"Stefan-Boltzmann constant sigma = {sigma_exact:.7e} W/(m^2*K^4) (Text: 5.670374e-8)")

    # Wien's Displacement Law derivation
    # d/dlambda [ 1 / (lambda^5 (exp(hc/(lambda*k_B*T)) - 1)) ] = 0
    # Let x = hc / (lambda*k_B*T).
    # d/dx [ x^5 / (exp(x) - 1) ] = 0 => 5(e^x - 1) - x e^x = 0 => x / (1 - e^-x) = 5
    def wien_func(x):
        return x / (1.0 - math.exp(-x)) - 5.0

    x_root = optimize.root_scalar(wien_func, bracket=[4.0, 6.0]).root
    b_wien = (h * c) / (x_root * k_B) # m*K
    b_wien_um_K = b_wien * 1e6
    print(f"Wien root x = {x_root:.8f}")
    print(f"Wien displacement constant b = {b_wien_um_K:.5f} um*K (Text: 2897.77 um*K)")

    # Test peak wavelengths:
    targets = [
        ("Earth Ambient", 300.0, 9.66),
        ("Smelter / Kiln", 750.0, 3.86),
        ("Refinery Gas Flare", 1400.0, 2.07),
        ("High-Intensity Explosion", 1800.0, 1.61)
    ]
    for name, T, expected_lambda in targets:
        l_max = b_wien_um_K / T
        print(f"  {name} (T={T}K): lambda_max = {l_max:.4f} um (Expected: {expected_lambda} um, Diff: {abs(l_max - expected_lambda):.4f})")

    # Planck's Law Spectral Radiance: B(lambda, T) in W / (m^2 * sr * um)
    # Note: B_lambda(lambda, T) = (2 h c^2) / (lambda^5 (exp(hc/(lambda k_B T)) - 1))
    # In units of W / (m^2 * sr * m), then multiply by 1e-6 to get W / (m^2 * sr * um).
    def planck_radiance(lam_um, T_K):
        lam_m = lam_um * 1e-6
        val = (2 * h * c**2) / (lam_m**5 * (math.exp((h * c) / (lam_m * k_B * T_K)) - 1.0))
        return val * 1e-6 # W / (m^2 * sr * um)

    def invert_planck(lam_um, L_val):
        lam_m = lam_um * 1e-6
        L_si = L_val * 1e6
        arg = 1.0 + (2 * h * c**2) / (L_si * lam_m**5)
        T = (h * c) / (lam_m * k_B * math.log(arg))
        return T

    # Quantitative Sub-Pixel Derivation Test
    # Flare Af = 10 m^2, Tf = 1400 K, VIIRS pixel = 375m x 375m = 140625 m^2, p = 10/140625 = 7.1111e-5, Tb = 300 K
    p = 10.0 / (375.0 * 375.0)
    # MWIR 3.9 um
    b_mwir_300 = planck_radiance(3.9, 300.0)
    b_mwir_1400 = planck_radiance(3.9, 1400.0)
    l_mwir = p * b_mwir_1400 + (1 - p) * b_mwir_300
    t_mwir_app = invert_planck(3.9, l_mwir)

    # LWIR 11.0 um
    b_lwir_300 = planck_radiance(11.0, 300.0)
    b_lwir_1400 = planck_radiance(11.0, 1400.0)
    l_lwir = p * b_lwir_1400 + (1 - p) * b_lwir_300
    t_lwir_app = invert_planck(11.0, l_lwir)

    delta_t = t_mwir_app - t_lwir_app

    print("\n--- Sub-Pixel Radiance Verification ---")
    print(f"p = {p:.6e} (Text: 7.11e-5)")
    print(f"MWIR (3.9 um): B(300K) = {b_mwir_300:.4f} W/(m^2*sr*um) (Text: 0.53)")
    print(f"MWIR (3.9 um): B(1400K) = {b_mwir_1400:.2f} W/(m^2*sr*um) (Text: 13,850)")
    print(f"Ratio B(1400K)/B(300K) = {b_mwir_1400/b_mwir_300:.1f}x (Text: 26,132x)")
    print(f"L_3.9 composite = {l_mwir:.4f} W/(m^2*sr*um) (Text: 1.515)")
    print(f"Apparent T_I4 = {t_mwir_app:.2f} K (Text: 338.4 K, Diff: {abs(t_mwir_app - 338.4):.2f} K)")
    print(f"LWIR (11.0 um): B(300K) = {b_lwir_300:.4f} W/(m^2*sr*um) (Text: 9.52)")
    print(f"LWIR (11.0 um): B(1400K) = {b_lwir_1400:.2f} W/(m^2*sr*um) (Text: 158.4)")
    print(f"Ratio B(1400K)/B(300K) = {b_lwir_1400/b_lwir_300:.1f}x (Text: 16.6x)")
    print(f"L_11.0 composite = {l_lwir:.4f} W/(m^2*sr*um) (Text: 9.531)")
    print(f"Apparent T_I5 = {t_lwir_app:.2f} K (Text: 300.1 K, Diff: {abs(t_lwir_app - 300.1):.2f} K)")
    print(f"Differential Delta T = {delta_t:.2f} K (Text: 38.3 K)")

    # MWIR vs LWIR Temperature Sensitivity Scaling d(ln B)/d(ln T)
    x_mwir = c2_um_K / (3.9 * 300.0)
    x_lwir = c2_um_K / (11.0 * 300.0)
    scale_mwir = x_mwir * math.exp(x_mwir) / (math.exp(x_mwir) - 1.0)
    scale_lwir = x_lwir * math.exp(x_lwir) / (math.exp(x_lwir) - 1.0)
    print(f"\n--- Temperature Exponent Scaling at T=300K ---")
    print(f"MWIR (3.9 um): x={x_mwir:.3f}, d(ln B)/d(ln T) = T^{scale_mwir:.2f} (Text: T^10 to T^12)")
    print(f"LWIR (11.0 um): x={x_lwir:.3f}, d(ln B)/d(ln T) = T^{scale_lwir:.2f} (Text: T^4 to T^5)")

def verify_spatial_temporal_math():
    print("\n==================================================")
    print("2. SPATIO-TEMPORAL FORMULATION VERIFICATION")
    print("==================================================")
    # 2.1 Spatial Distance Decay W_spatial
    sigma_s = 375.0
    d_max = 1500.0
    def w_spatial(dist):
        if dist > d_max:
            return 0.0
        return math.exp(- (dist**2) / (2.0 * sigma_s**2))

    distances = [0.0, 187.5, 375.0, 750.0, 1125.0, 1500.0, 1501.0]
    print("W_spatial evaluations:")
    for d in distances:
        print(f"  dist = {d:6.1f} m -> W_spatial = {w_spatial(d):.6f}")

    # 2.2 Temporal Recurrence Score S_recurrence
    half_life_days = 30.0
    lam = math.log(2) / half_life_days
    print(f"\nRecurrence decay constant lambda = {lam:.6f} day^-1 (Text: 0.0231)")

    t_now = 90.0
    times = np.arange(0, 90, 1.0)
    weights = np.exp(-lam * (t_now - times))
    denom = np.sum(weights)

    flare_detections = np.random.RandomState(42).binomial(1, 0.8, size=len(times))
    s_flare = np.sum(flare_detections * weights) / denom

    transient_old = np.zeros(len(times))
    transient_old[5] = 1.0
    s_trans_old = np.sum(transient_old * weights) / denom

    recent_fire = np.zeros(len(times))
    recent_fire[-3:] = 1.0
    s_recent = np.sum(recent_fire * weights) / denom

    print(f"  S_recurrence (Persistent Flare ~80% detection) = {s_flare:.4f}")
    print(f"  S_recurrence (Transient anomaly 85 days ago)   = {s_trans_old:.6f}")
    print(f"  S_recurrence (Recent wildfire active 3 days)   = {s_recent:.4f}")

    # 2.3 Diurnal Persistence Index (DPI)
    def compute_dpi(mu_night, mu_day, sigma_frp, mu_frp, eps=1e-5):
        ratio = mu_night / (mu_day + eps)
        stability = 1.0 - (sigma_frp / (mu_frp + eps))
        return ratio * stability

    dpi_flare = compute_dpi(mu_night=25.0, mu_day=26.0, sigma_frp=5.0, mu_frp=25.5)
    dpi_stubble = compute_dpi(mu_night=0.5, mu_day=18.0, sigma_frp=12.0, mu_frp=10.0)
    dpi_wildfire = compute_dpi(mu_night=15.0, mu_day=45.0, sigma_frp=40.0, mu_frp=30.0)

    print(f"\nDPI (Flare)    = {dpi_flare:.4f} (Expected: [0.70, 1.00])")
    print(f"DPI (Stubble)  = {dpi_stubble:.4f} (Expected: <= 0.15)")
    print(f"DPI (Wildfire) = {dpi_wildfire:.4f} (Expected: <= 0.15)")

    # 2.4 ST-DBSCAN Dual Metric
    eps_s = 750.0 # meters
    eps_t = 72.0  # hours
    def d_st(d_geo, d_time_h):
        return (d_geo / eps_s) + (d_time_h / eps_t)

    print(f"\nST-DBSCAN Distance at (d=375m, dt=36h): D_ST = {d_st(375, 36):.2f} (<= 1.0 -> Core Match)")
    print(f"ST-DBSCAN Distance at (d=800m, dt=10h): D_ST = {d_st(800, 10):.2f} (> 1.0 -> Out of cluster)")

    # 2.5 H3 k-ring quadratic formula verification:
    for k in [1, 2, 3, 4, 5]:
        n_cells = 1 + 3 * k * (k + 1)
        print(f"H3 k-ring size for k={k}: {n_cells} cells (Text: k=1->7, k=2->19, k=3->37)")

def verify_ml_loss_and_gradients():
    print("\n==================================================")
    print("3. MACHINE LEARNING LOSS & GRADIENT VERIFICATION")
    print("==================================================")
    gamma = 2.0
    alpha = 1.0

    def softmax(z):
        ez = np.exp(z - np.max(z))
        return ez / np.sum(ez)

    def focal_loss_vector(z, target_idx, gamma=2.0, alpha=1.0):
        p = softmax(z)
        p_t = p[target_idx]
        loss = - alpha * ((1.0 - p_t)**gamma) * np.log(max(p_t, 1e-15))
        return loss

    z_test = np.array([2.5, 0.5, -1.0, 0.2, -0.5, 1.0])
    target = 0
    p_test = softmax(z_test)
    eps = 1e-6
    grad_num = np.zeros_like(z_test)
    for i in range(len(z_test)):
        z_plus = z_test.copy()
        z_plus[i] += eps
        z_minus = z_test.copy()
        z_minus[i] -= eps
        l_plus = focal_loss_vector(z_plus, target, gamma, alpha)
        l_minus = focal_loss_vector(z_minus, target, gamma, alpha)
        grad_num[i] = (l_plus - l_minus) / (2 * eps)

    p_t = p_test[target]
    term_bracket = 1.0 + gamma * p_t * math.log(p_t) / (1.0 - p_t)
    grad_ana_target = alpha * ((1.0 - p_t)**gamma) * (p_t - 1.0) * term_bracket

    print(f"Target class p_t = {p_t:.6f}")
    print(f"Target logit grad - Numerical:  {grad_num[target]:.8f}")
    print(f"Target logit grad - Analytical: {grad_ana_target:.8f}")
    print(f"Absolute Error: {abs(grad_num[target] - grad_ana_target):.2e}")

    for m in range(len(z_test)):
        if m != target:
            p_m = p_test[m]
            grad_ana_m = alpha * p_m * ((1.0 - p_t)**gamma) * (1.0 - gamma * p_t * math.log(p_t) / (1.0 - p_t))
            print(f"  Non-target logit {m} (p={p_m:.4f}) - Num: {grad_num[m]:.8f}, Ana: {grad_ana_m:.8f}, Diff: {abs(grad_num[m] - grad_ana_m):.2e}")

    p_easy = 0.98
    p_hard = 0.01
    suppr_easy = (1.0 - p_easy)**2
    suppr_hard = (1.0 - p_hard)**2
    print(f"\nGradient modulation factor (1 - p)^2:")
    print(f"  Easy sample (p=0.98): (1 - 0.98)^2 = {suppr_easy:.6f} (Suppression: {1.0/suppr_easy:.1f}x, Text: 2500x)")
    print(f"  Hard sample (p=0.01): (1 - 0.01)^2 = {suppr_hard:.6f} (Text: 0.9801)")

def verify_evaluation_metrics():
    print("\n==================================================")
    print("4. EVALUATION METRIC VERIFICATION")
    print("==================================================")
    C = np.array([
        [450,   2,   5,   1,   0,   2],
        [  1,  48,   0,   1,   0,   0],
        [  8,   0, 890,  12,   0,  10],
        [  2,   1,  15, 380,   0,   2],
        [  0,   0,   0,   0,  20,   0],
        [  3,   0,   8,   1,   0, 138]
    ])
    N = np.sum(C)
    K = C.shape[0]

    f1_list = []
    iou_list = []
    for k in range(K):
        tp = C[k, k]
        fp = np.sum(C[:, k]) - tp
        fn = np.sum(C[k, :]) - tp
        f1_k = (2.0 * tp) / (2.0 * tp + fp + fn)
        iou_k = tp / (tp + fp + fn)
        f1_list.append(f1_k)
        iou_list.append(iou_k)
        print(f"Class {k+1}: TP={tp:3d}, FP={fp:2d}, FN={fn:2d} -> F1={f1_k:.4f}, IoU={iou_k:.4f}")

    f1_macro = np.mean(f1_list)
    print(f"\nMacro-F1 = {f1_macro:.4f}")

    po = np.trace(C) / N
    row_sums = np.sum(C, axis=1)
    col_sums = np.sum(C, axis=0)
    pe = np.sum(row_sums * col_sums) / (N**2)
    kappa = (po - pe) / (1.0 - pe)
    print(f"Observed Agreement p_o = {po:.4f}")
    print(f"Expected Agreement p_e = {pe:.4f}")
    print(f"Cohen's Kappa kappa    = {kappa:.4f}")

    np.random.seed(42)
    y_true_onehot = np.zeros((N, K))
    probs = np.zeros((N, K))
    idx = 0
    for true_c in range(K):
        for pred_c in range(K):
            count = C[true_c, pred_c]
            for _ in range(count):
                y_true_onehot[idx, true_c] = 1.0
                p_vec = np.ones(K) * 0.02
                p_vec[pred_c] += 0.88
                p_vec /= np.sum(p_vec)
                probs[idx] = p_vec
                idx += 1

    bs_total = np.mean(np.sum((probs - y_true_onehot)**2, axis=1))
    print(f"Mean Multi-Class Brier Score BS = {bs_total:.4f}")

def verify_xai_and_shap():
    print("\n==================================================")
    print("5. EXPLAINABLE AI & SHAP VERIFICATION")
    print("==================================================")
    T = 1200
    L = 63
    D = 7
    ops_treeshap = T * L * (D**2)
    ops_exact_shap = 2**38

    print(f"Exact Shapley combinations: 2^38 = {ops_exact_shap:.6e} evaluations (Text: 2.74e11)")
    print(f"TreeSHAP operations: O(T * L * D^2) = {T} * {L} * {D**2} = {ops_treeshap:,} ops (Text: sub-5ms)")

    e_fx = -3.20
    phi_beta = [3.85, 3.12, 1.90, 1.40]
    f_x_beta = e_fx + sum(phi_beta)
    p_beta = 1.0 / (1.0 + math.exp(-f_x_beta))
    print(f"\nIncident Beta (Disaster):")
    print(f"  Base E[f(x)] = {e_fx:.2f}")
    print(f"  Sum of phi_i = {sum(phi_beta):.2f}")
    print(f"  Final log-odds f(x) = {f_x_beta:.2f} (Text: +7.07)")
    print(f"  Calibrated Probability P(Emergency) = {p_beta * 100:.2f}% (Text: 99.1%)")

    phi_alpha_flare = [2.45, 1.90, -1.95, -0.10, 1.32]
    f_x_alpha = e_fx + sum(phi_alpha_flare)
    p_alpha = 1.0 / (1.0 + math.exp(-f_x_alpha))
    print(f"Incident Alpha (Flare):")
    print(f"  Net log-odds f(x) = {f_x_alpha:.2f}")
    print(f"  Calibrated Probability P(Flare) = {p_alpha * 100:.2f}% (Text: 97.4%)")

if __name__ == "__main__":
    verify_thermal_physics()
    verify_spatial_temporal_math()
    verify_ml_loss_and_gradients()
    verify_evaluation_metrics()
    verify_xai_and_shap()
