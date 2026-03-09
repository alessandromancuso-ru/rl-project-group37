# ENVIRONMENT

# DEFINE STATES
states = []
TERMINAL_STATE = "TERMINAL_STATE"

# Calculate all possible states
for player_sum in range(12, 22):
    for dealer_card in range(1, 11):
        for usable_ace in [False, True]:
            states.append((player_sum, dealer_card, usable_ace))
            
states.append(TERMINAL_STATE)

# Card possible values
card_values = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# Define card probabilities (1/3 for all cards except cards with value 10 which have 4/10 prob)
card_probs = card_probs = [
    1/13, 1/13, 1/13, 1/13, 1/13,
    1/13, 1/13, 1/13, 1/13,
    4/13
]

# Actions
HIT_ACTION = 1
STICK_ACTION = 0

# UTILITY FUCNTIONS

def is_state_terminal(state):
    return state == TERMINAL_STATE

def hit_transitions_from_state(state):
    """
    Simulate all the possible hit transitions given a state
    It takes a state as input:
    - player_sum: the sum of the cards the player have in hand
    - dealer_card: the dealer card visible
    - usable_ace: the presence of a usable ace in hand of player
    Returns the transitions list
    """
    
    if is_state_terminal(state):
        return [(1.0, TERMINAL_STATE, 0.0, True)]

    player_sum, dealer_card, usable_ace = state

    transitions = []

    # Loop su tutte le possibili carte
    for card_value, prob in zip(card_values, card_probs):
        new_player_sum = player_sum + card_value
        new_usable_ace = usable_ace
        
        # Case ace
        if card_value == 1:     
            # Check if the ace can assume value 11
            if new_player_sum + 10 <= 21:
                # In case take the 11
                new_player_sum += 10
                new_usable_ace = True
                
        if new_player_sum > 21 and new_usable_ace:
            new_player_sum -= 10
            new_usable_ace = False
        
        #  Bust case: the sum is > 21, next state is terminal and reward: -1
        if new_player_sum > 21:
            next_state = TERMINAL_STATE
            reward = -1
            done = True
        else:
            next_state = (new_player_sum, dealer_card, new_usable_ace)
            reward = 0
            done = False
        
        transitions.append((prob, next_state, reward, done))

    return transitions

def stick_transitions_from_state(state):
    """
    Simulate all the possible stick transitions given a state
    It takes a state as input:
    - player_sum: the sum of the cards the player have in hand
    - dealer_card: the dealer card visible
    - usable_ace: the presence of a usable ace in hand of player
    Returns the transitions list
    """
    
    if is_state_terminal(state):
            return [(1.0, TERMINAL_STATE, 0.0, True)]

    expected_reward = expected_reward_stick(state)
    
    return [(1.0, TERMINAL_STATE, expected_reward, True)]

def dealer_initial_states(dealer_card):
    """
    Returns a list of (prob_init, start_sum, usable_ace_dealer)
    coherent with the dealer's visible card.
    """
    
    dealer_initial_states = []
    dealer_sum = 0
    
    for hidden_card, prob_hidden in zip(card_values, card_probs):
        dealer_sum = dealer_card + hidden_card
        usable_ace_dealer = False
        
        if dealer_card == 1 or hidden_card == 1:
            dealer_sum += 10
            usable_ace_dealer = True
    
        dealer_initial_states.append((prob_hidden, dealer_sum, usable_ace_dealer))
        
    return dealer_initial_states
    
def expected_reward_stick(state):
    player_sum, dealer_card, usable_ace_player = state
    
    total_reward = 0
    
    for initial_prob, dealer_sum, dealer_usable_ace in dealer_initial_states(dealer_card):
        outcomes = dealer_outcomes(dealer_sum, dealer_usable_ace)

        for outcome_prob, dealer_final_sum, dealer_bust in outcomes:
            # reward dal punto di vista del player
            if dealer_bust:
                reward = 1
            else:
                if dealer_final_sum > player_sum:
                    reward = -1
                elif dealer_final_sum < player_sum:
                    reward = 1
                else:
                    reward = 0

            total_reward += initial_prob * outcome_prob * reward

    return total_reward

def dealer_outcomes(dealer_sum, dealer_usable_ace):
    """
    Simulate all the possible outcomes of the dealer:
    - dealer_sum: the sum of the cards the dealer have in hand
    - dealer_usable_ace: the presence of a usable ace in hand of the dealer
    Returns the outcomes list: (prob, sum_finale, bust_flag)
    """
    
    if dealer_sum >= 17:
        # Stick
        bust_flag = dealer_sum > 21
        return [(1.0, dealer_sum, bust_flag)]
        
    if dealer_sum < 17:
        outcomes = []
        
        # Hit
        for card_value, prob in zip(card_values, card_probs):
            new_dealer_sum = dealer_sum + card_value
            new_dealer_usable_ace = dealer_usable_ace
            # new_prob = prob * past_prob
            
            # Case ace
            if card_value == 1:     
                # Check if the ace can assume value 11
                if new_dealer_sum + 10 <= 21:
                    # In case take the 11
                    new_dealer_sum += 10
                    new_dealer_usable_ace = True
                    
            if new_dealer_sum > 21 and new_dealer_usable_ace:
                new_dealer_sum -= 10
                new_dealer_usable_ace = False
            
            if new_dealer_sum >= 17:
                bust_flag = new_dealer_sum > 21
                outcomes.append((prob, new_dealer_sum, bust_flag))
            else:
                next_outcomes = dealer_outcomes(new_dealer_sum, new_dealer_usable_ace)
                
                for next_prob, next_sum, next_bust_flag in next_outcomes:
                    total_prob = prob * next_prob
                    outcomes.append((total_prob, next_sum, next_bust_flag))
                
        return outcomes


# POLICY
# Define the policy
# Simple policy: player_sum >= 20 Stick, otherwise Hit

policy = {}

for state in states:
    
    if is_state_terminal(state):
        continue
    
    player_sum, dealer_card, usable_ace = state
    if player_sum >= 20:
        policy[state] = 0  # Stick
    else:
        policy[state] = 1  # Hit

# TRANSITION MODEL
# P(s', r | s, a)
#
# It maps each state-action pair to all possible outcomes
# with their associated probabilities.

P = {}

for state in states:
    P[state] = {STICK_ACTION: [], HIT_ACTION: []}

for state in states:
    P[state][STICK_ACTION] = stick_transitions_from_state(state) # Stick transitions
    P[state][HIT_ACTION] = hit_transitions_from_state(state)    # Hit transitions