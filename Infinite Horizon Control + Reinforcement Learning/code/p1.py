from collections import OrderedDict


def motion_model_prob2(si, action, bet):
    if not si:
        return []
    if si >= 30:
        return [(si, 1.0)]
    return [(si + 10, 0.7), (si - 10, 0.3)]


def motion_model(si, action, bet):
    if not si:
        return []
    if si >= 30:
        return [(si, 1.0)]
    if action == "Red":
        return [(si + bet, 0.7), (si - bet, 0.3)]
    if action == "Black":
        return [(si + 10 * bet, 0.2), (si - bet, 0.8)]


def policy_iteration(states, policy, actions, motion_model):
    # Step 1: Initialize states
    V = OrderedDict()
    for state in states:
        if state >= 30:
            V[state] = -state
        else:
            V[state] = 0

    while True:
        # Step 2: Policy Evaluation
        delta = float('inf')
        threshhold = 1e-10
        while delta >= threshhold:
            delta = 0
            for s in states:
                old_v = V[s]
                V[s] = sum([prob * V[s_prime] for s_prime, prob in motion_model(s, *policy[s])])
                delta = max(delta, abs(V[s] - old_v))

        # Step 3: Policy Improvement
        is_policy_stable = True
        for s in states:
            old_action = policy[s]
            min_val = float('inf')
            for a in actions:
                for money_bet in range(10, s+1, 10):
                    val = sum([prob * V[s_prime] for s_prime, prob in motion_model(s, a, money_bet)])
                    if val < min_val:
                        min_val, policy[s] = val, (a, money_bet)
            if old_action != policy[s]:
                is_policy_stable = False

        if is_policy_stable:
            break

    return V


def main():
    # Initialize parameters in problem 2
    states = [0, 10, 20, 30, 40, 110, 120, 220]  # The unit is 1k
    policy = {s: ("Red", 10) for s in states}   # state: (action, money to bet)
    policy[0] = ("None", 0)
    actions = ["Red", "Black"]

    V = policy_iteration(states, policy, actions, motion_model)
    for k, val in V.items():
        print(f"V[{k}] = {val}")

    for s, (a, bet) in policy.items():
        print(f"policy[{s}]: action = [{a}], [{bet*1000}]")

    print(to_table(V, policy))


def to_table(V, policy):
    for k, val in V.items():
        print(f"{k} & {val*1000} & {policy[k][0]} + {policy[k][1] * 1000} \\\\\n\\hline")


if __name__ == "__main__":
    main()