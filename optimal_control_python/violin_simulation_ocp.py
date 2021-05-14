import numpy as np
import pickle
from bioptim import OptimalControlProgram

from violin_ocp import Violin, ViolinString, ViolinOcp, Bow, BowTrajectory, BowPosition
from bioptim import Solver


if __name__ == "__main__":
    model_name = "WuViolin"
    violin = Violin(model_name, ViolinString.E)
    bow = Bow(model_name)

    # --- Solve the program --- #
    n_shoot_per_cycle = 30
    cycle_time = 1
    n_cycles = 1
    solver = Solver.IPOPT
    n_threads = 2
    ocp = ViolinOcp(
        model_path=f"../models/{model_name}.bioMod",
        violin=violin,
        bow=bow,
        n_cycles=n_cycles,
        bow_starting=BowPosition.TIP,
        init_file=None,
        use_muscles=False,
        fatigable=True,
        time_per_cycle=cycle_time,
        n_shooting_per_cycle=n_shoot_per_cycle,
        solver=solver,
        n_threads=n_threads
    )
    # ocp, sol = ViolinOcp.load("results/5_cycles_34_muscles/2021_3_12.bo")

    lim = bow.hair_limits if ocp.bow_starting == BowPosition.FROG else [bow.hair_limits[1], bow.hair_limits[0]]
    bow_trajectory = BowTrajectory(lim, ocp.n_shooting_per_cycle + 1)
    bow_target = np.tile(bow_trajectory.target[:, :-1], ocp.n_cycles)
    bow_target = np.concatenate((bow_target, bow_trajectory.target[:, -1][:, np.newaxis]), axis=1)
    ocp.set_bow_target_objective(bow_target)

    from time import time
    t = time()
    sol = ocp.solve(
        show_online_optim=True,
        solver_options={"max_iter": 1000, "hessian_approximation": "limited-memory", "linear_solver": "ma57"},
    )
    t2 = time() - t
    sol.print()
    sol.animate(show_meshes=False)

    # Save results without stand alone
    ocp.save(sol, False)
    ocp_load, sol_load = OptimalControlProgram.load("results/5_cycles_with_fatigue.bo")
    #sol_load.animate()
    sol_load.graphs()

    # Save results with stand alone
    ocp.save(sol, True)
    with open(f"results/5_cycles_with_fatigue_sa.bo", "rb") as file:
        states, controls, parameters = pickle.load(file)

    # Print time to optimize
    from time import time
    print(f"Graphing time = {t2 - sol.time_to_optimize}")
    print(f"Graphing time = {sol.time_to_optimize}")

    # Print results for humerus_right_rot_z
    import matplotlib.pyplot as plt

    q_humerus_right_rot_z = states['q'][6]
    qdot_humerus_right_rot_z = states['qdot'][6]
    tau_neg_humerus_right_rot_z = controls['tau'][6]
    tau_pos_humerus_right_rot_z = controls['tau'][19]
    ma_neg_humerus_right_rot_z = states['fatigue'][37]
    mr_neg_humerus_right_rot_z = states['fatigue'][38]
    mf_neg_humerus_right_rot_z = states['fatigue'][39]
    ma_pos_humerus_right_rot_z = states['fatigue'][40]
    mr_pos_humerus_right_rot_z = states['fatigue'][41]
    mf_pos_humerus_right_rot_z = states['fatigue'][42]

    x = np.array([i for i in range(0, 31)])

    plt.figure()
    plt.plot(x, q_humerus_right_rot_z)
    plt.title("q_humerus_right_rot_z")

    plt.figure()
    plt.plot(x, qdot_humerus_right_rot_z)
    plt.title("qdot_humerus_right_rot_z")

    plt.figure()
    plt.plot(x, tau_pos_humerus_right_rot_z)
    plt.plot(x, tau_neg_humerus_right_rot_z)
    plt.title("tau_humerus_right_rot_z")

    plt.figure()
    plt.plot(x, ma_pos_humerus_right_rot_z, color='green')
    plt.plot(x, mr_pos_humerus_right_rot_z, color='red')
    plt.plot(x, -mf_pos_humerus_right_rot_z, color='blue')
    plt.plot(x, -ma_neg_humerus_right_rot_z, color='green')
    plt.plot(x, -mr_neg_humerus_right_rot_z, color='red')
    plt.plot(x, mf_neg_humerus_right_rot_z, color='blue')
    plt.title("fatigue_humerus_right_rot_z")

    plt.plot()

    # Root mean square (or use scikit-learn rmse ?)
    from scipy import integrate as intg
    from statistics import mean
    from math import sqrt

    q_humerus_right_rot_z = states['q'][6]
    qdot_humerus_right_rot_z = states['qdot'][6]

    def rmse(list1: list, list2: list):
        from statistics import mean
        len_list = len(list1)
        x = np.array([i for i in range(0, len_list)])
        I1 = intg.trapz(list1, x)
        I2 = intg.trapz(list2, x)
        sub_I = []
        for i in range(0, len_list):
            sub_I.append((I1[i] - I2[i]) * (I1[i] - I2[i]))
        mean = mean(sub_I)
        rmse = sqrt(mean)
        return rmse

    rmse = rmse(q_humerus_right_rot_z, qdot_humerus_right_rot_z)
