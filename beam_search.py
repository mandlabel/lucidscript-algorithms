def beam_search(start_state, evaluate, generate_successors, beam_width=3, is_goal=None):
    """
    Beam Search algorithm.
    
    Parameters:
    - start_state: The initial state.
    - evaluate: A function to evaluate the quality of a state.
    - generate_successors: A function to generate possible successors from a state.
    - beam_width: The number of top states to keep after each iteration.
    - is_goal: A function to check if a state satisfies the goal condition.
    
    Returns:
    - best_state: The optimal state found.
    """
    current_states = [start_state]  # Start with the initial state

    while True:
        # Generate all successors for the current states
        successors = []
        for state in current_states:
            successors.extend(generate_successors(state))

        # If no successors are found, terminate
        if not successors:
            break

        # Evaluate all successors
        scored_successors = [(state, evaluate(state)) for state in successors]

        # Sort by score in descending order
        scored_successors.sort(key=lambda x: x[1], reverse=True)

        # Keep the top 'beam_width' states
        current_states = [state for state, score in scored_successors[:beam_width]]

        # If any state satisfies the goal condition, return it
        if is_goal and any(is_goal(state) for state in current_states):
            return next(state for state in current_states if is_goal(state))

    # Return the best state found
    return max(current_states, key=evaluate)


# Example problem: Finding the smallest number divisible by 5 using transformations
def evaluate(state):
    # The closer to zero, the better the evaluation (negative for minimization)
    return -state

def generate_successors(state):
    # Generate successors by adding or subtracting 1
    return [state + 1, state - 1]

def is_goal(state):
    # Goal is finding a number divisible by 5
    return state % 5 == 0

# Initial state
start_state = 12

# Beam search execution
best_state = beam_search(start_state, evaluate, generate_successors, beam_width=3, is_goal=is_goal)
print("Best state found:", best_state)
