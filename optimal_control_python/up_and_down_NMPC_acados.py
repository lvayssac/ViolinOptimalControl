import biorbd
import numpy as np
from bioptim import ObjectiveFcn, ObjectiveList, Node, Solver

from violin_ocp import Bow, Violin, Optim
from violin_ocp.bow_trajectory import generate_bow_trajectory, curve_integral


# Options
biorbd_model_path = "../models/BrasViolon.bioMod"
begin_at_first_iter = True
regenerate_bow_trajectory = False
ns_all_optim = 600
ns_one_cycle = 150  # size of the up_and_down gesture
window_time = 1 / 8  # duration of the window
window_len = 80  # size of NMPC window
violin = Violin("E")
bow = Bow("frog")

# Aliases
biorbd_model = biorbd.Model(biorbd_model_path)
n_q = biorbd_model.nbQ()
n_qdot = biorbd_model.nbQdot()
n_tau = biorbd_model.nbGeneralizedTorque()
n_muscles = biorbd_model.nbMuscles()

if regenerate_bow_trajectory:
    bow_target_param = generate_bow_trajectory(200)
    np.save("bow_target_param", bow_target_param)
else:
    bow_target_param = np.load("bow_target_param.npy")
frame_to_init_from = window_len

Q_est_acados = np.zeros((n_q, ns_all_optim))
X_est_acados = np.zeros((n_q + n_qdot, ns_all_optim))
Qdot_est_acados = np.zeros((n_qdot, ns_all_optim))
U_est_acados = np.zeros((n_tau, ns_all_optim))
if begin_at_first_iter:
    # Initial guess and bounds
    x0 = np.array(violin.q()[bow.side] + [0] * n_qdot)

    x_init = np.tile(np.array(violin.q()[bow.side] + [0] * n_qdot)[:, np.newaxis], window_len + 1)
    u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis], window_len)
else:
    X_est_init = np.load("x_est.npy")[:, : window_len + 1]
    U_est_init = np.load("u_est.npy")[:, : window_len + 1]
    x_init = X_est_init[:, -(window_len + 1) :]
    x0 = x_init[:, 0]
    u_init = U_est_init[:, -window_len:]

# position initiale de l'ocp
ocp, x_bounds = Optim.prepare_generic_ocp(
    biorbd_model_path=biorbd_model_path,
    nb_shooting=window_len,
    final_time=window_time,
    x_init=x_init,
    u_init=u_init,
    x0=x0,
    is_acados=True,
    use_sx=True,
)

t = np.linspace(0, 2, ns_one_cycle)
target_curve = curve_integral(bow_target_param, t)
q_target = np.ndarray((n_q, window_len + 1))
Nmax = ns_all_optim + window_len
target = np.ndarray((Nmax,))
T = np.ndarray((Nmax,))
for i in range(Nmax):
    a = i % ns_one_cycle
    T[i] = t[a]
target = curve_integral(bow_target_param, T)
shift = 1


# Init from known position
# ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter_acados.bo")
# data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
# x_init, u_init, X_out, U_out, x_bounds, u = warm_start_nmpc(sol=sol_load, ocp=ocp,
# window_len=window_len, n_q=n_q, n_qdot=n_qdot, n_tau=n_tau, biorbd_model=biorbd_model,
#                                                             acados=True, shift=1)
# U_est_acados[:, :U_est_init.shape[1]] = U_est_init
# X_est_acados[:, :X_est_init.shape[1]] = X_est_init
# Q_est_acados[:, :X_est_init.shape[1]]=X_est_init[:10]
# Qdot_est_acados[:, :X_est_init.shape[1]] = X_est_init[10:]

# for i in range(frame_to_init_from, 200):
for i in range(0, 150):
    print(f"iteration:{i}")
    if i < 300:
        q_target[bow.hair_idx, :] = target[i * shift : window_len + (i * shift) + 1]
        new_objectives = ObjectiveList()
        new_objectives.add(
            ObjectiveFcn.Lagrange.SUPERIMPOSE_MARKERS,
            node=Node.ALL,
            weight=100000,
            first_marker_idx=Bow.contact_marker,
            second_marker_idx=violin.bridge_marker,
            list_index=1,
        )
        new_objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_ALL_CONTROLS, node=Node.ALL, weight=10, list_index=2)
        new_objectives.add(ObjectiveFcn.Lagrange.MINIMIZE_STATE, node=Node.ALL, weight=10, list_index=3)
        new_objectives.add(
            ObjectiveFcn.Lagrange.TRACK_STATE,
            node=Node.ALL,
            weight=100000,
            target=q_target[bow.hair_idx : bow.hair_idx + 1, :],
            index=bow.hair_idx,
            list_index=4,
        )
        ocp.update_objectives(new_objectives)
    else:
        q_target[bow.hair_idx, :] = target[i - 40 * shift : window_len + (i - 40 * shift) + 1]
        if target[i] < -0.45:  # but : mettre des poids plus lourds aux extremums de la target pour que les extremums
            weight = 1500  # ne soient pas dépassés par le poids des des autres valeurs "itermédiaires" de la target
        if target[i] > -0.17:  # qui sont majoritaire dans la fenêtre
            weight = 1500
        else:
            weight = 1000
        Optim.update_target_objective(ocp, bow, q_target, weight)
    if i == 0:
        sol = ocp.solve(
            show_online_optim=False,
            solver=Solver.ACADOS,
            solver_options={"nlp_solver_max_iter": 1000},
        )
    else:
        sol = ocp.solve(
            show_online_optim=False,
            solver=Solver.ACADOS,
        )
    x_init, u_init, X_out, U_out, x_bounds = Optim.warm_start_nmpc(
        sol=sol,
        ocp=ocp,
        window_len=window_len,
        n_q=n_q,
        n_qdot=n_qdot,
        n_tau=n_tau,
        biorbd_model=biorbd_model,
        acados=True,
        shift=shift,
    )
    Q_est_acados[:, i] = X_out[:10]
    X_est_acados[:, i] = X_out
    Qdot_est_acados[:, i] = X_out[10:]
    U_est_acados[:, i] = U_out

    # ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

np.save("Q_est_acados", Q_est_acados)
np.save("X_est_acados", X_est_acados)
np.save("Qdot_est_acados", Qdot_est_acados)
np.save("U_est_acados", U_est_acados)
out = np.load("U_est_acados.npy")