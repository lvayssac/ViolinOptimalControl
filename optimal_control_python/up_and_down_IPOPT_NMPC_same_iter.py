import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_up_and_down_bow_target import generate_up_and_down_bow_target
from optimal_control_python.generate_up_and_down_bow_target import curve_integral
from optimal_control_python.utils import Bow, Violin
from optimal_control_python.utils_functions import prepare_generic_ocp, warm_start_nmpc, define_new_objectives, \
    display_graphics_X_est, display_X_est, compare_target, warm_start_nmpc_same_iter
from bioptim import OptimalControlProgram, Data, ObjectiveList, Objective, Node



if __name__ == "__main__":
    # Parameters
    biorbd_model_path = "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    final_time = 1/8  # duration of the
    nb_shooting_pts_window = 15  # size of NMPC window
    ns_tot_up_and_down = 150 # size of the up_and_down gesture

    violin = Violin("E")
    bow = Bow("frog")

    # np.save("bow_target_param", generate_up_and_down_bow_target(200))
    bow_target_param = np.load("bow_target_param.npy")
    frame_to_init_from = nb_shooting_pts_window+1
    nb_shooting_pts_all_optim = 300

    X_est = np.zeros((n_qdot + n_q , nb_shooting_pts_all_optim))
    U_est = np.zeros((n_tau, nb_shooting_pts_all_optim))
    Q_est = np.zeros((n_q , nb_shooting_pts_all_optim))
    Qdot_est = np.zeros((n_qdot , nb_shooting_pts_all_optim))

    begin_at_first_iter = True
    if begin_at_first_iter == True :
        x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)

        x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                         nb_shooting_pts_window+1)
        u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                         nb_shooting_pts_window)
    else:
        X_est_init = np.load('X_est.npy')[:, :frame_to_init_from+1]
        U_est_init = np.load('U_est.npy')[:, :frame_to_init_from+1]
        x_init = X_est_init[:, -(nb_shooting_pts_window+1):]
        x0 = x_init[:, 0]
        u_init = U_est_init[:, -nb_shooting_pts_window:]

    # position initiale de l'ocp
    ocp, x_bounds = prepare_generic_ocp(
        biorbd_model_path=biorbd_model_path,
        number_shooting_points=nb_shooting_pts_window,
        final_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x0=x0,
        acados=False,
        useSX=False,
    )


    t = np.linspace(0, 2, ns_tot_up_and_down)
    target_curve = curve_integral(bow_target_param, t)
    q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
    Nmax = nb_shooting_pts_all_optim+50
    target = np.ndarray(Nmax)
    T = np.ndarray((Nmax))
    for i in range(Nmax):
        a=i % ns_tot_up_and_down
        T[i]=t[a]
    target = curve_integral(bow_target_param, T)

    shift = 1


    # Init from known position
    # ocp_load, sol_load = OptimalControlProgram.load(f"saved_iterations/{frame_to_init_from-1}_iter.bo")
    # data_sol_prev = Data.get_data(ocp_load, sol_load, concatenate=False)
    # x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x = warm_start_nmpc(sol=sol_load, ocp=ocp,
    # nb_shooting_pts_window=nb_shooting_pts_window, n_q=n_q, n_qdot=n_qdot, n_tau=n_tau, biorbd_model=biorbd_model, acados=False, shift=1)
    # U_est[:, :U_est_init.shape[1]] = U_est_init
    # X_est[:, :X_est_init.shape[1]] = X_est_init



    for i in range(0, 30):
        print(f"iteration:{i}")
        q_target[bow.hair_idx, :] = target[0 * shift: nb_shooting_pts_window + (0 * shift) + 1]# target[1 * shift: nb_shooting_pts_window + (1 * shift) + 1]

        define_new_objectives(weight=1000, ocp=ocp, q_target=q_target, bow=bow)
        sol = ocp.solve(
            show_online_optim=False,
            solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10 ** (-10),
                            "bound_frac": 10 ** (-10), "warm_start_init_point":"yes",
                            "warm_start_bound_push" : 10 ** (-16), "warm_start_bound_frac" : 10 ** (-16),
                            "nlp_scaling_method": "none", "warm_start_mult_bound_push": 10 ** (-16),
                            # "warm_start_slack_bound_push": 10 ** (-16)," warm_start_slack_bound_frac":10 ** (-16),
                            }
        )
        # x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x= warm_start_nmpc(sol=sol, ocp=ocp,
        #                                                             nb_shooting_pts_window=nb_shooting_pts_window,
        #                                                             n_q = n_q, n_qdot=n_qdot, n_tau=n_tau,
        #                                                             biorbd_model=biorbd_model,
        #                                                             acados=False, shift=shift) #, lam_g, lam_x
        x_init, u_init, X_out, U_out, x_bounds, u, lam_g, lam_x= warm_start_nmpc_same_iter(sol=sol, ocp=ocp, biorbd_model=biorbd_model)
        # warm_start_nmpc(sol, ocp, nb_shooting_pts_window, n_q, n_qdot, n_tau, biorbd_model, acados, shift=1)
        # sol['lam_g'] = lam_g # pas utile en same iter...
        # sol['lam_x'] = lam_x
        Q_est[:, i] = X_out[:10]
        X_est[:, i] = X_out
        Qdot_est[:, i] = X_out[10:]
        U_est[:, i] = U_out

        ocp.save(sol, f"saved_iterations/{i}_iter_acados")  # you don't have to specify the extension ".bo"

    np.save("Q_est__s", Q_est)
    np.save("X_est__s", X_est)
    np.save("Qdot_est__s", Qdot_est)
    np.save("U_est__s", U_est)
    np.save("U_est__s", U_est)


