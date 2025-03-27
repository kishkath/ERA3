# Reinforcement learning Part-1: GridWorld Value Iteration Assignment

This project demonstrates a simple reinforcement learning problem using a 4x4 GridWorld and the value iteration algorithm to compute optimal state values.

## Overview

- **Environment:** A 4x4 grid where each cell is a state.
- **Actions:** Up, down, left, or right.
- **Rewards:** Each move costs -1; reaching the terminal state (bottom-right, state `(3,3)`) gives a reward of 0.
- **Terminal State:** The bottom-right corner; once reached, its value remains 0.
- **Objective:** Compute state values using the Bellman equation until the maximum change is less than \(10^{-4}\).

## Reinforcement Learning Concepts

- **Value Iteration:** Uses the Bellman equation to calculate optimal state values, representing the maximum expected cumulative reward from each state.
- **Bellman Equation:** Recursively breaks down the value of a state into immediate reward and the discounted value of next states.
- **Policy vs. Plan:**
  - **Policy:** A mapping from states to actions.
  - **Planning:** Using value iteration to compute state values and derive the optimal policy.
- **MDP Framework:** Models the environment where the next state depends only on the current state and chosen action.

## Project Structure

- **`reinforcement_learning_part1.ipynb`**: Notebook with the implementation.
- **`README.md`**: This file.

## Code Explanation

1. **Environment Setup:**  
   - Grid size: 4x4 (`N = 4`).
   - Discount factor: 1.0 (`gamma = 1.0`).
   - Convergence threshold: \(10^{-4}\).

2. **Actions:**  
   - Moves represented as: Up `(-1, 0)`, Down `(1, 0)`, Left `(0, -1)`, Right `(0, 1)`.

3. **Terminal State:**  
   - The terminal state is `(3, 3)`, which is not updated during iterations.

4. **State Transitions:**  
   - The function checks for grid boundaries; if a move is invalid, the state remains unchanged.

5. **Value Iteration:**  
   - Initialize \(V(s)\) to 0.
   - For each non-terminal state, update:
     \[
     V(s) = \frac{1}{4} \sum_{a}\left(-1 + \gamma \cdot V(s')\right)
     \]
   - Iterate until maximum change < \(10^{-4}\).

6. **Output:**  
   - A 4x4 matrix showing the computed state values:
     ```
     [[-59.42367735 -57.42387125 -54.2813141  -51.71012579]
      [-57.42387125 -54.56699476 -49.71029394 -45.13926711]
      [-54.2813141  -49.71029394 -40.85391609 -29.99766609]
      [-51.71012579 -45.13926711 -29.99766609   0.        ]]
     ```

## Conclusion

This assignment shows how value iteration can compute the optimal state values in a simple grid-based RL task. It demonstrates the use of the Bellman equation, the MDP framework, and the difference between planning (computing state values) and executing a policy.

Feel free to reach out with any questions or feedback!
