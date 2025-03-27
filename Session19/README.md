# GridWorld Value Iteration Assignment

This project demonstrates a simple reinforcement learning (RL) problem using a 4x4 GridWorld and the value iteration algorithm. The goal is to compute the optimal state values for the grid using the Bellman equation.

## Overview

In this assignment, we consider:
- **Environment:** A 4x4 grid representing a world where an agent navigates from a start to a goal.
- **States:** Each cell in the grid is a state.
- **Actions:** The agent can move up, down, left, or right.
- **Rewards:** Each move incurs a reward of **-1**, while the terminal state (bottom-right corner) gives a reward of **0**.
- **Terminal State:** The bottom-right corner (state `(3,3)` for a 4x4 grid) is the terminal state. Once reached, the episode ends, and its value remains fixed at 0.
- **Objective:** Compute the optimal state values by iteratively applying the Bellman equation until convergence (i.e., when the maximum change in value across all states is less than \(10^{-4}\)).

## Reinforcement Learning Context

This assignment is based on fundamental concepts of reinforcement learning:
- **Value Iteration:** A dynamic programming method that computes optimal state values (\(V^*(s)\)) using the Bellman equation. These values represent the maximum expected cumulative reward obtainable from each state.
- **Bellman Equation:** A recursive update that decomposes the value of a state into immediate reward plus the discounted value of successor states.
- **Policy vs. Plan:**
  - **Policy:** A mapping from states to actions (the agent's behavior).
  - **Planning (Value Iteration):** Computing state values to eventually derive an optimal policy.
- **Markov Decision Process (MDP):** The framework used to model this problem, where future states depend only on the current state and chosen action (the Markov property).

## Project Structure

- **`reinforcement_learning_part1.ipynb`**  
  Contains the implementation of the value iteration algorithm for the 4x4 GridWorld.
  
- **`README.md`**  
  This file, which explains the assignment, concepts, and how to run the code.

## Code Explanation

1. **Environment Setup:**
   - The grid is set to 4x4 (`N = 4`).
   - The discount factor is set to 1.0 (`gamma = 1.0`), meaning future rewards are not discounted.
   - A convergence threshold (`threshold = 1e-4`) determines when the algorithm stops iterating.

2. **Actions:**
   - Defined as tuples representing moves: Up (`(-1, 0)`), Down (`(1, 0)`), Left (`(0, -1)`), and Right (`(0, 1)`).

3. **Terminal State:**
   - The terminal state is `(3, 3)`. A helper function `is_terminal(state)` checks if a state is terminal. The terminal state's value is not updated during iterations since it remains at 0.

4. **State Transitions:**
   - The function `get_next_states(state)` computes the next state for each action, considering grid boundaries. If an action would take the agent out of bounds, the agent remains in the same state.

5. **Value Iteration:**
   - The algorithm initializes the value function \(V(s)\) to zeros.
   - For each non-terminal state, it applies the Bellman update:
     \[
     V(s) = \frac{1}{4} \sum_{a} \left(-1 + \gamma \cdot V(s')\right)
     \]
     where the summation is over all four possible actions.
   - The algorithm iterates until the maximum change in any state's value is less than \(10^{-4}\).

6. **Output:**
   - The final value function is printed as a 4x4 matrix, showing the computed values for each state.

## OUTPUT

  ** Output:**
   - A matrix of state values obtained:
     ```
     [[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
     [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
     [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
     [-51.71012579 -45.13926711 -29.99766609   0.        ]]
     ```
   - This output reflects the computed optimal state values, with the highest (zero) at the terminal state.

## Conclusion

This project illustrates how value iteration can be used in a simple grid-based reinforcement learning task. By understanding and implementing the Bellman equation, you learn how to evaluate the desirability of states and derive an optimal strategy for navigating environments.

For further study, consider:
- Extending this model to more complex environments.
- Extracting an optimal policy from the computed state values.
- Experimenting with different rewards and discount factors.

Feel free to reach out with any questions or feedback!
