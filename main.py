'''
discount factor = 0.9
두 location에 있을 수 있는 차 개수 = state = 0, 1, ..., 20
1에서 2로 move한 차 개수 = -5, -4, ..., 4, 5
  (단, 움직인 후 두 lcoation에 있는 차 개수가 0 미만 또는 20 초과면 안됨)
rent된 차 개수 = min(rent request, state + move)
  (rent 후에도 차 개수가 0과 20 사이에 있기 위함)
return된 차 개수 = min (return request, 20 - (state + move - request))
  (return 후에도 차 개수가 0과 20 사이에 있기 위함)
'''
MAX_CAR = 20
MAX_MOVE = 5
REWARD_RENTAL = 10.0
REWARD_MOVE = -2.0

# poisson lambda
RENT1, RETURN1 = 3, 3
RENT2, RETURN2 = 4, 2

# discount factor
GAMMA = 0.9

# repeat policy evaluation until delta < THETA
THETA = 0.01

import numpy as np
from scipy.stats import poisson
import matplotlib.pyplot as plt
import time, sys, os


def plot(nparray, title='', path='plot.png'):
    im = plt.imshow(nparray, origin='lower')
    plt.colorbar(im)
    plt.title(title)
    plt.xlabel('Location 2')
    plt.ylabel('Location 1')
    #plt.show()
    plt.savefig(path, dpi=100)
    plt.clf()


poisson_dict = dict()
states = list()


def init_poisson_dict(poisson_dict):
    for mu in set((RENT1, RENT2, RETURN1, RETURN2)):
        for i in range(MAX_CAR + 1):
            poisson_dict[mu * MAX_CAR + i] = poisson.pmf(i, mu)

def poisson_sf(poisson_dict, n, mu):
    try:
        v = poisson_dict[mu * MAX_CAR * 10 + n]
    except KeyError:
        v = poisson.sf(n, mu)   # sf = 1 - cdf
        poisson_dict[mu * MAX_CAR * 10 + n] = v
    return v


def init():
    global reward_rental, p_1, p_2
    print("Initializing...")
    st = time.time()
    poisson_dict = dict()
    init_poisson_dict(poisson_dict)
    for s1 in range(0, MAX_CAR + 1):
        for s2 in range(0, MAX_CAR + 1):
            reward_rental_ = 0.
            for rent1 in range(0, s1 + 1):
                if rent1 == s1:
                    p_rent1 = poisson_sf(poisson_dict, rent1 - 1, RENT1)
                else:
                    p_rent1 = poisson_dict[rent1 + RENT1 * MAX_CAR]

                for rent2 in range(0, s2 + 1):
                    if rent2 == s2:
                        p_rent2 = poisson_sf(poisson_dict, rent2 - 1, RENT2)
                    else:
                        p_rent2 = poisson_dict[rent2 + RENT2 * MAX_CAR]

                    for return1 in range(0, MAX_CAR - (s1 - rent1) + 1):
                        if return1 == MAX_CAR - (s1 - rent1):
                            p_return1 = poisson_sf(poisson_dict, return1 - 1, RETURN1)
                        else:
                            p_return1 = poisson_dict[return1 + RETURN1 * MAX_CAR]
                        
                        for return2 in range(0, MAX_CAR - (s2 - rent2) + 1):
                            if return2 == MAX_CAR - (s2 - rent2):
                                p_return2 = poisson_sf(poisson_dict, return2 - 1, RETURN2)
                            else:
                                p_return2 = poisson_dict[return2 + RETURN2 * MAX_CAR]
                            p = p_rent1 * p_return1 * p_rent2 * p_return2

                            reward_rental_ += p * REWARD_RENTAL * (rent1 + rent2)
            reward_rental[s1, s2] = reward_rental_
    
    for s1 in range(0, MAX_CAR + 1):
        for rent1 in range(0, s1 + 1):
            if rent1 == s1:
                p_rent1 = poisson_sf(poisson_dict, rent1 - 1, RENT1)
            else:
                p_rent1 = poisson_dict[rent1 + RENT1 * MAX_CAR]
            
            for return1 in range(0, MAX_CAR - (s1 - rent1) + 1):
                if return1 == MAX_CAR - (s1 - rent1):
                    p_return1 = poisson_sf(poisson_dict, return1 - 1, RETURN1)
                else:
                    p_return1 = poisson_dict[return1 + RETURN1 * MAX_CAR]
                    
                p_1[s1, return1 - rent1 + s1] += p_rent1 * p_return1
    
    for s2 in range(0, MAX_CAR + 1):
        for rent2 in range(0, s2 + 1):
            if rent2 == s2:
                p_rent2 = poisson_sf(poisson_dict, rent2 - 1, RENT2)
            else:
                p_rent2 = poisson_dict[rent2 + RENT2 * MAX_CAR]

            for return2 in range(0, MAX_CAR - (s2 - rent2) + 1):
                if return2 == MAX_CAR - (s2 - rent2):
                    p_return2 = poisson_sf(poisson_dict, return2 - 1, RETURN2)
                else:
                    p_return2 = poisson_dict[return2 + RETURN2 * MAX_CAR]
            
                p_2[s2, return2 - rent2 + s2] += p_rent2 * p_return2
    print(f"Initialization Done! Time: {time.time() - st:.1f}")
    #print(f"{reward_rental}\n\n{p_1}\n\n{p_2}")


def calculate_value(s1, s2, mv, v_old):
    global p_1, p_2, reward_rental

    s1_, s2_ = s1 - mv, s2 + mv
    p = np.outer(p_1[s1_, :], p_2[s2_, :])
    p_dot_v = np.multiply(p, v_old)

    return reward_rental[s1_, s2_] + REWARD_MOVE * abs(mv) + GAMMA * np.sum(p_dot_v)


def policy_evaluation(v_old, policy):
    iteration = 1
    st_total = time.time()
    while True:
        st = time.time()
        delta = 0.0
        v_new = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
        for s1 in range(0, MAX_CAR + 1):
            for s2 in range(0, MAX_CAR + 1):
                mv = policy[s1, s2]     # number of cars moved from 1 to 2
                value_new = calculate_value(s1, s2, mv, v_old)
                v_new[s1, s2] = value_new
                delta = max(delta, abs(value_new - v_old[s1, s2]))
        #print(f"PE {iter:2d} delta: {delta:6.3f} time: {time.time() - st:.1f}")
        if delta < THETA:
            break
        else:
            v_old = v_new
            iteration += 1
    print(f"Policy Evaluation finished! Iteration: {iteration}")
    return v_new


def policy_improvement(value):
    p_new = np.zeros((MAX_CAR + 1, MAX_CAR + 1), dtype=np.intc)
    for s1 in range(0, MAX_CAR + 1):
        for s2 in range(0, MAX_CAR + 1):
            value_max = -100.0

            for mv in range(-min(MAX_MOVE, s2, MAX_CAR - s1), min(MAX_MOVE, s1, MAX_CAR - s2) + 1):
                value_current = calculate_value(s1, s2, mv, value)
                if value_max < value_current:
                    value_max, policy_argmax = value_current, mv
            p_new[s1, s2] = policy_argmax
    print("Policy Improvement finished!")
    return p_new


if __name__ == '__main__':
    np.set_printoptions(precision=3, suppress=True, linewidth=150)
    # initialize
    value = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
    policy = np.zeros((MAX_CAR + 1, MAX_CAR + 1), dtype=np.intc)

    reward_rental = np.empty((MAX_CAR + 1, MAX_CAR + 1))
    p_1 = np.zeros((MAX_CAR + 1, MAX_CAR + 1))
    p_2 = np.zeros((MAX_CAR + 1, MAX_CAR + 1))

    init()

    policy_stable = False
    iteration = 1
    while not policy_stable:
        print(f"=========Iteration {iteration}=========")
        value = policy_evaluation(value, policy)
        value_saved = value[:]
        policy_new = policy_improvement(value)

        policy_stable = np.all(policy_new == policy)
        policy = policy_new

        plot(value, f"value_{iteration}", f"value_{iteration}.png")
        plot(policy, f"policy_{iteration}", f"policy_{iteration}.png")

        print(f"Policy stable: {policy_stable}")
        iteration += 1