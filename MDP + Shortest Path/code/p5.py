import numpy as np
from tqdm import tqdm
from collections import Counter, OrderedDict, defaultdict
import matplotlib.pyplot as plt

'''
Time = (#R + #P + #S)
'''

class Move(int):
    R = 0
    P = 1
    S = 2

class Judge(int):
    WIN = set([(Move.R, Move.S), (Move.S, Move.P), (Move.P, Move.R)])
    LOSE = set([(Move.S, Move.R), (Move.P, Move.S), (Move.R, Move.P)])
    DRAW = set([(Move.R, Move.R), (Move.S, Move.S), (Move.P, Move.P)])

ALL_MOVES = [Move.R, Move.P, Move.S]

preference_model = np.load('preference_model.npy')
# preference_model_old = np.load('preference_model_old.npy')
policy = np.load('policy.npy')
value_star = np.load('value_star.npy')

def generate_combinations(T):
    res = []
    for r in range(0, T + 1):
        for p in range(0, T + 1):
            for s in range(0, T + 1):
                if r + p + s == T:
                    res.append((r,p,s))
    return res


def p_x_given_f(X: Move, F:Move):
    return 0.5 if X == F else 0.25


def previous(r, p, s):
    res = OrderedDict()
    if r - 1 >= 0:
        res[Move.R] = (r-1, p, s)
    if p - 1 >= 0:
        res[Move.P] = (r, p-1, s)
    if s - 1 >= 0:
        res[Move.S] = (r, p, s-1)
    return res

def next_case(r, p, s):
    res = OrderedDict()
    res[Move.R] = (r + 1, p, s)
    res[Move.P] = (r, p + 1, s)
    res[Move.S] = (r, p, s + 1)
    return res


def p_preference_given_history_old(T=105):

    # (F, R, P, S)
    preference_model = np.zeros(shape=(3, T + 1, T + 1, T + 1))
    preference_model[Move.R][0][0][0] = 1/3
    preference_model[Move.P][0][0][0] = 1/3
    preference_model[Move.S][0][0][0] = 1/3

    for t in tqdm(range(1, T + 1)):
        for r, p, s in generate_combinations(t):
            prev = previous(r, p, s)
            for F in ALL_MOVES:
                for next_move, (prev_r, prev_p, prev_s) in prev.items():
                    preference_model[F][r][p][s] += (p_x_given_f(next_move, F) * preference_model[F][prev_r][prev_p][prev_s]) / np.sum([(p_x_given_f(next_move, f) * preference_model[f][prev_r][prev_p][prev_s]) for f in ALL_MOVES])
    np.save('preference_model.npy', preference_model)
    return preference_model

def p_preference_given_history(T=105):
    # (F, R, P, S)
    preference_model = np.zeros(shape=(3, T + 1, T + 1, T + 1))
    preference_model[Move.R][0][0][0] = 1 / 3
    preference_model[Move.P][0][0][0] = 1 / 3
    preference_model[Move.S][0][0][0] = 1 / 3

    for t in tqdm(range(1, T + 1)):
        for r, p, s in generate_combinations(t):
            # prev = previous(r, p, s)
            for F in ALL_MOVES:
                preference_model[F][r][p][s] = p_history_given_f(F, r, p, s) / (3 * np.sum([p_history_given_f(f, r, p, s) for f in ALL_MOVES]))
            #     for next_move, (prev_r, prev_p, prev_s) in prev.items():
            #         preference_model[F][r][p][s] += (p_x_given_f(next_move, F) * preference_model[F][prev_r][prev_p][prev_s]) / np.sum([(p_x_given_f(next_move, f) * preference_model[f][prev_r][prev_p][prev_s]) for f in ALL_MOVES])
    np.save('preference_model.npy', preference_model)
    return preference_model


    pass

def p_history_given_f(f, r, p, s):
    p = 1.0
    for move, count in enumerate([r, p, s]):
        p *= 0.5 ** count if f == move else 0.25 ** count
    return p

def p_next_given_history(x_next, Tr, Tp, Ts):
    return np.sum([p_x_given_f(x_next, F) * preference_model[F][Tr][Tp][Ts] for F in ALL_MOVES])


def judge(case):
    if case in Judge.WIN:
        return 1
    elif case in Judge.LOSE:
        return -1
    elif case in Judge.DRAW:
        return 0
    else:
        raise Exception("Invalid case")


def dynamic_programming(T=100):

    # (R, P, S, SD)
    policy = np.zeros(shape=(T + 1, T + 1, T + 1, 2*T+1))
    value_star = np.zeros(shape=(T + 1, T + 1, T + 1, 2*T+1))

    def sd_to_idx(sd):
        return sd + T

    # Set final stage
    for r, p, s in generate_combinations(T):
        for sd in range(-T, T + 1):
            value_star[r][p][s][sd_to_idx(sd)] = sd

    # dynamic programming
    for t in tqdm(range(T - 1, -1, -1)):
        for r, p, s in generate_combinations(t):
            next_status = next_case(r, p, s).items()
            for sd in range(-t, t+1):
                # Find policy to maximize value_star[r][p][s][sd]
                best_move, best_value = None, float('-inf')
                for my_move in ALL_MOVES:
                    q_value = 0.0
                    for opp_move, (next_r, next_p, next_s) in next_status:
                        q_value += p_next_given_history(opp_move, r, p, s) * value_star[next_r][next_p][next_s][sd_to_idx(sd + judge((my_move, opp_move)))]
                    if q_value > best_value:
                        best_value = q_value
                        best_move = my_move
                policy[r][p][s][sd] = best_move
                value_star[r][p][s][sd] = best_value
    np.save('policy.npy', policy)
    np.save('value_star.npy', value_star)

# def test_optimal():
#
#     # my_moves = np.random.choice(np.array([Move.R, Move.P, Move.S]), size=100, p=[1/3, 1/3, 1/3]).tolist()
#     # my_moves = [0, 1, 2] * 100
#     # my_moves = np.random.choice()
#     # print(Counter(opp_moves))
#
#     scores = []
#     for _ in range(100):
#         opp_moves = np.random.choice(np.array([Move.R, Move.P, Move.S]), size=100, p=[0.5, 0.25, 0.25]).tolist()
#         opp_history = dict()
#         opp_history[Move.R] = 0
#         opp_history[Move.P] = 0
#         opp_history[Move.S] = 0
#         score = 0
#         for i, opp_move in enumerate(opp_moves):
#             # my_move = my_moves[i]
#             my_move = int(policy[opp_history[Move.R]][opp_history[Move.P]][opp_history[Move.S]][score] + 1) % 3
#             # my_move = int(policy[opp_history[Move.R]][opp_history[Move.P]][opp_history[Move.S]][score])
#             # print(my_move, opp_move, score, value_star[opp_history[Move.R]][opp_history[Move.P]][opp_history[Move.S]][score], opp_history)
#             score += judge((my_move, opp_move))
#             opp_history[opp_move] += 1
#         scores.append(score)
#         # print(f"Score = {score}")
#     print(np.average(np.array(scores)))

class Strategy(int):
    DETERMINISTIC = 0
    STOCHASTIC = 1
    OPTIMAL = 2

def test(game_num, match_num=50):

    # （strategy, match, game）
    scores = np.zeros(shape=(3, match_num, game_num))
    for match in range(match_num):
        opp_moves = np.random.choice(np.array([Move.R, Move.P, Move.S]), size=100, p=[0.25, 0.5, 0.25]).tolist()
        deterministic_moves = np.array([Move.R, Move.P, Move.S] * 40)
        stochastic_moves = np.random.choice(np.array([Move.R, Move.P, Move.S]), size=100, p=[1/3, 1/3, 1/3]).tolist()
        opp_history = dict()
        opp_history[Move.R] = 0
        opp_history[Move.P] = 0
        opp_history[Move.S] = 0
        score = 0
        for game_round, opp_move in enumerate(opp_moves):
            scores[Strategy.DETERMINISTIC][match][game_round] += judge((deterministic_moves[game_round], opp_moves[game_round]))
            scores[Strategy.STOCHASTIC][match][game_round] += judge((stochastic_moves[game_round], opp_moves[game_round]))
            opt_move = int(policy[opp_history[Move.R]][opp_history[Move.P]][opp_history[Move.S]][score] + 2) % 3
            score += judge((opt_move, opp_move))
            scores[Strategy.OPTIMAL][match][game_round] = score
            opp_history[opp_move] += 1

    X = np.array(list(range(game_num)))
    score_mean = np.zeros(shape=(3, game_num))
    for t in range(game_num):
        score_mean[Strategy.DETERMINISTIC][t] = np.mean(scores[Strategy.DETERMINISTIC,:,t])
        score_mean[Strategy.STOCHASTIC][t] = np.mean(scores[Strategy.STOCHASTIC,:,t])
        score_mean[Strategy.OPTIMAL][t] = np.mean(scores[Strategy.OPTIMAL,:,t])
    plt.plot(X, score_mean[Strategy.DETERMINISTIC], linewidth=1, label="deterministic")
    plt.plot(X, score_mean[Strategy.STOCHASTIC], linewidth=1, label="stochastic")
    plt.plot(X, score_mean[Strategy.OPTIMAL], linewidth=1, label="optimal")
    plt.xlabel("Number of games")
    plt.ylabel("Mean value of score differential")
    plt.legend()
    plt.savefig("mean")
    plt.show()

    score_std = np.zeros(shape=(3, game_num))
    for t in range(game_num):
        score_std[Strategy.DETERMINISTIC][t] = np.std(scores[Strategy.DETERMINISTIC,:,t])
        score_std[Strategy.STOCHASTIC][t] = np.std(scores[Strategy.STOCHASTIC,:,t])
        score_std[Strategy.OPTIMAL][t] = np.std(scores[Strategy.OPTIMAL,:,t])
    plt.plot(X, score_std[Strategy.DETERMINISTIC], linewidth=1, label="deterministic")
    plt.plot(X, score_std[Strategy.STOCHASTIC], linewidth=1, label="stochastic")
    plt.plot(X, score_std[Strategy.OPTIMAL], linewidth=1, label="optimal")
    plt.xlabel("Number of games")
    plt.ylabel("Standard diviation of score differential")
    plt.legend()
    plt.savefig("std")
    plt.show()

if __name__=='__main__':
    # Uncomment the first two lines to generate needed npy files. This may take about 35 minutes
    # p_preference_given_history()
    # dynamic_programming(T=100)
    test(game_num=100)
