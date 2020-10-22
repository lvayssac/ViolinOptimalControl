import biorbd
import numpy as np
import matplotlib
from optimal_control_python.generate_up_and_down_bow_target import generate_up_and_down_bow_target
from optimal_control_python.generate_up_and_down_bow_target import curve_integral


from bioptim import (
    OptimalControlProgram,
    Objective,
    ObjectiveList,
    DynamicsType,
    DynamicsTypeOption,
    Constraint,
    ConstraintList,
    BoundsOption,
    QAndQDotBounds,
    InitialGuessOption,
    Instant,
    InterpolationType,
    Data,
)
from optimal_control_python.utils import Bow, Violin


def prepare_ocp(biorbd_model_path, number_shooting_points, final_time, x_init, u_init, x0, ):
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_tau = biorbd_model.nbGeneralizedTorque()
    tau_min, tau_max, tau_init = -100, 100, 0

    objective_functions = ObjectiveList()
    objective_functions.add(Objective.Lagrange.MINIMIZE_TORQUE, idx=0)
    objective_functions.add(
        Objective.Lagrange.ALIGN_SEGMENT_WITH_CUSTOM_RT,
        weight=100,
        segment_idx=Bow.segment_idx,
        rt_idx=violin.rt_on_string,
        idx=1
    )


    dynamics = DynamicsTypeOption(DynamicsType.TORQUE_DRIVEN)

    x_bounds = BoundsOption(QAndQDotBounds(biorbd_model))
    x_bounds[:, 0] = x0
    x_init = InitialGuessOption(x_init, interpolation=InterpolationType.EACH_FRAME)

    u_bounds = BoundsOption([[tau_min] * n_tau, [tau_max] * n_tau])
    u_init = InitialGuessOption(u_init, interpolation=InterpolationType.EACH_FRAME)

    new_constraints = ConstraintList()
    for j in range(1, 5):
        new_constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            min_bound=0,
                            max_bound=0,
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker)
    for j in range(5, nb_shooting_pts_window + 1):
        new_constraints.add(Constraint.ALIGN_MARKERS,
                            instant=j,
                            # min_bound=-1, #-10**(j-14) donne 25 itérations
                            # max_bound=1, # (j-4)/10 donne 21 itérations
                            first_marker_idx=Bow.contact_marker,
                            second_marker_idx=violin.bridge_marker)

    return OptimalControlProgram(
        biorbd_model,
        dynamics,
        number_shooting_points,
        final_time,
        x_init,
        u_init,
        x_bounds,
        u_bounds,
        objective_functions=objective_functions,
        constraints=new_constraints,
    ), x_bounds



def warm_start_mhe(sol):
    data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
    q = data_sol_prev[0]["q"]
    dq = data_sol_prev[0]["q_dot"]
    u = data_sol_prev[1]["tau"]
    x = np.vstack([q, dq])
    X_out = x[:, 0]
    U_out = u[:, 0]
    x_init = x
    u_init = u[:, :-1]  # [:, 1:]  # discard oldest estimate of the window
    return x_init, u_init, X_out, U_out

    #
    # X0 = np.hstack((x[:, 1:], np.tile(x[:, [-1]], 1)))  # discard oldest estimate of the window, duplicates youngest
    # U0 = u[:, 1:]  # discard oldest estimate of the window
    # X_out = x[:, 0]
    # return X0, U0, X_out

if __name__ == "__main__":
    biorbd_model_path = "/home/carla/Documents/Programmation/ViolinOptimalControl/models/BrasViolon.bioMod"
    biorbd_model = biorbd.Model(biorbd_model_path)
    n_q = biorbd_model.nbQ()
    n_qdot = biorbd_model.nbQdot()
    n_tau = biorbd_model.nbGeneralizedTorque()
    n_muscles = biorbd_model.nbMuscles()
    final_time = 1/3  # duration of the simulation
    nb_shooting_pts_window = 15  # size of NMPC window
    ns_tot = nb_shooting_pts_window*2 # size of the entire optimization
    time_tot=2/3

    violin = Violin("E")
    bow = Bow("frog")
    bow_target = np.ndarray((n_q, nb_shooting_pts_window + 1))


    x0 = np.array(violin.initial_position()[bow.side] + [0] * n_qdot)
    x_init = np.tile(np.array(violin.initial_position()[bow.side] + [0] * n_qdot)[:, np.newaxis],
                     nb_shooting_pts_window+1)
    # x_init[0, :] = 0.07
    u_init = np.tile(np.array([0.5] * n_tau)[:, np.newaxis],
                     nb_shooting_pts_window)

    X_est = np.zeros((n_q * 2, nb_shooting_pts_window+1))
    U_est = np.zeros((n_tau * 2, nb_shooting_pts_window))

    bow_target_param = generate_up_and_down_bow_target(200)
    # position initiale de l'ocp

    ocp, x_bounds = prepare_ocp(
        biorbd_model_path=biorbd_model_path,
        number_shooting_points=nb_shooting_pts_window,
        final_time=final_time,
        x_init=x_init,
        u_init=u_init,
        x0=x0,
        # q_ref=q_ref,
    )

    for i in range(ns_tot):

        n_points = 200
        final_time
        t = np.linspace(0, final_time, nb_shooting_pts_window + 1)
        q_target = np.ndarray((n_q, nb_shooting_pts_window + 1))
        q_target[bow.hair_idx, :] = curve_integral(bow_target_param, t)

        new_objectives = ObjectiveList()

        new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE, instant=Instant.ALL, state_idx=bow.hair_idx,
                           weight=1,
                           idx=3) # permet de réduire le nombre d'itérations avant la convergence
        new_objectives.add(
            Objective.Lagrange.TRACK_STATE, instant=Instant.ALL, weight=100, target=q_target, states_idx=bow.hair_idx, idx=4
        )
        # new_objectives.add(Objective.Lagrange.MINIMIZE_TORQUE_DERIVATIVE, instant=Instant.ALL, state_idx=bow.hair_idx,
        #                    # weight=1,
        #                    idx=4)  # rajoute des itérations et ne semble riuen changer au mouvement...
        ocp.update_objectives(new_objectives)

        sol = ocp.solve(
            show_online_optim=False,
            solver_options={"max_iter": 1000, "hessian_approximation": "exact", "bound_push": 10**(-10), "bound_frac": 10**(-10)}  #, "bound_push": 10**(-10), "bound_frac": 10**(-10)}
        )

        data_sol_prev = Data.get_data(ocp, sol, concatenate=False)
        x_init, u_init, X_out, U_out = warm_start_mhe(sol)

        ocp, x_bounds = prepare_ocp(
            biorbd_model_path=biorbd_model_path,
            number_shooting_points=nb_shooting_pts_window,
            final_time=final_time,
            x_init=x_init,
            u_init=u_init,
            x0=x0,
        )


        sol = ocp.solve(
            show_online_optim=False,
            solver_options={"max_iter": 0, "hessian_approximation": "exact", "bound_push": 10**(-10), "bound_frac": 10**(-10)}  #, "bound_push": 10**(-10), "bound_frac": 10**(-10)}
        )


        matplotlib.pyplot.figure(1)

        data_sol = Data.get_data(ocp, sol, concatenate=False)

        matplotlib.pyplot.suptitle('Q et x_init')
        for idx in range(10): # idx = degré de liberté
            matplotlib.pyplot.subplot(2, 5, int(idx+1))

            matplotlib.pyplot.plot(data_sol[0]["q"][idx], color="blue") # degré de liberté idx à tous les noeuds
            matplotlib.pyplot.plot(data_sol_prev[0]["q"][idx], color="yellow")
            matplotlib.pyplot.title(f"dof {idx}")
            matplotlib.pyplot.plot(x_init[idx, :], color="red") # degré de liberté idx à tous les noeuds
            matplotlib.pyplot.plot(x_bounds.min[idx, :], color="green")
            matplotlib.pyplot.plot(x_bounds.max[idx, :], color="green")

        matplotlib.pyplot.figure(2)
        matplotlib.pyplot.suptitle('Qdot et x_init')

        for idx in range(10): # idx = degré de liberté
            matplotlib.pyplot.subplot(2, 5, int(idx+1))
            matplotlib.pyplot.title(f"dof {idx}")
            matplotlib.pyplot.plot(data_sol[0]["q_dot"][idx], color="blue") # degré de liberté idx à tous les noeuds
            matplotlib.pyplot.plot(data_sol_prev[0]["q_dot"][idx], color="yellow")
            matplotlib.pyplot.plot(x_init[idx+n_q, :], color="red") # degré de liberté idx à tous les noeuds
            matplotlib.pyplot.plot(x_bounds.min[idx+n_q, :], color="green")
            matplotlib.pyplot.plot(x_bounds.max[idx+n_q, :], color="green")
        matplotlib.pyplot.show()


        matplotlib.pyplot.figure(3)
        matplotlib.pyplot.suptitle('tau et u_init')
        for idx in range(10):  # idx = degré de liberté
            matplotlib.pyplot.subplot(2, 5, int(idx+1))
            matplotlib.pyplot.title(f"dof {idx}")
            matplotlib.pyplot.plot(data_sol[1]["tau"][idx], color="blue") # degré de liberté idx à tous les noeuds
            matplotlib.pyplot.plot(u_init[idx, :], color="red") # degré de liberté idx à tous les noeuds
        matplotlib.pyplot.show()



        print(f"NUMERO DE LA FENETRE : {i}")
        data_sol = Data.get_data(ocp, sol, concatenate=False)

        x_init, u_init, x0, u0 = warm_start_mhe(data_sol)

        X_est = x0
        U_est = u0

    # --- Show results --- #
    ocp.save_get_data(sol, "up_and_down_NMPC")

    np.save("results", X_est)





