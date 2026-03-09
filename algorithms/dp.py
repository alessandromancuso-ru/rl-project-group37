import numpy as np
from env.env_blackjack import P, TERMINAL_STATE, STICK_ACTION, HIT_ACTION, is_state_terminal

def q_value(state, action, P, V, discount = 1.0):
    """Calculate Q(s,a) using model P and values V."""
    q = 0.0
    for prob, next_state, reward, done in P[state][action]:
        q += prob * (reward + discount * V[next_state])
    return q

def policy_iteration(policy, P, theta=0.5, discount=1.0):
    
    # Initialization
    V = {s: 0.0 for s in P.keys()}
    V[TERMINAL_STATE] = 0.0
    
    # print(V)
    
    def policy_evaluation():
        nonlocal V
        
        while True:
            delta = 0.0
            
            for state in P.keys():
                
                if state == TERMINAL_STATE:
                    continue
                
                old_value = V[state]
                
                # Deterministic action based on plocy
                action = policy[state]
                # Possibile transitions from state with deterministic action
                transitions = P[state][action]
                
                expected_value = 0.0
                
                # Σ_{s′,r} p(s′,r | s, π(s)) [ r + γ V(s′) ]
                for prob, next_state, reward, done in transitions:
                    expected_value += prob * (reward + discount * V[next_state])
                    
                V[state] = expected_value
                delta = max(delta, abs(old_value - V[state]))  # Δ ← max(Δ, |v − V(s)|)
                
            if delta < theta:
                break
            
        return V

    V = policy_evaluation()
    
    def policy_improvement():
        policy_stable = True
        
        for state in P.keys():
            if state == TERMINAL_STATE:
                continue
        
            old_action = policy[state]
            
            qStick = q_value(state, 0, P, V, discount)
            qHit = q_value(state, 1, P, V, discount)
            
            if qStick > qHit:
                best_action = 0
            else:
                best_action = 1
            
            policy[state] = best_action
                
            if best_action != old_action:
                policy_stable = False
                
        return policy, policy_stable              
    
    while True:
        V = policy_evaluation()
        policy, stable_policy = policy_improvement()
        
        if stable_policy:
            break

    return V, policy

def value_iteration(P, states, theta=1e-4, discount=1.0):
    
    actions = [STICK_ACTION, HIT_ACTION]
    
    # Initialization
    V = {s: 0.0 for s in P.keys()}
    V[TERMINAL_STATE] = 0.0
    
    while True:
        delta = 0
        
        for state in states:
            
            # If the state is terminal continue (we cannot determine the best action for a terminal state)
            if is_state_terminal(state):
                continue
            
            v_old = V[state]
            
            max_action = -np.inf
            
            for action in actions:
                # Calculate q(s, a) using q_value
                action_value = q_value(state, action, P, V, discount)
                # Take the best action
                if action_value > max_action:
                    max_action = action_value
             
            V[state] = max_action
                
            delta = max(delta, abs(v_old-V[state]))
        
        if delta < theta:
            break

    # Establish and return a deterministic policy
    policy = {}
    
    for state in states:
        
        if is_state_terminal(state):
            continue
        
        q_value_stick = q_value(state, STICK_ACTION, P, V, discount)
        q_value_hit = q_value(state, HIT_ACTION, P, V, discount)
        
        if q_value_stick > q_value_hit:
            policy[state] = STICK_ACTION
        else:
            policy[state] = HIT_ACTION
            
    return V, policy
                
            
            
            
            
            
    