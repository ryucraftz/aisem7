import random
import pickle
from collections import defaultdict, deque

# -------------------------
# Tic-Tac-Toe Environment
# -------------------------
class TicTacToeEnv:
    def __init__(self):
        # board cells: 0 empty, 1 agent (X), 2 opponent (O)
        self.reset()

    def reset(self):
        self.board = [0] * 9
        self.current_player = 1  # by default agent (1) starts; caller can change
        self.done = False
        self.winner = None
        return self._get_state()

    def _get_state(self):
        # state as string for hashing in Q-table
        return ''.join(str(s) for s in self.board)

    def available_actions(self):
        return [i for i, v in enumerate(self.board) if v == 0]

    def step(self, action, player):
        """
        Place player's mark at action (0..8).
        Returns: (next_state, reward, done, info)
        reward is from the perspective of the agent (player 1).
        If player==1 (agent) and they win, reward +1.
        If player==2 (opponent) and opponent wins, reward -1.
        Draw -> reward 0
        Non-terminal -> reward 0
        """
        if self.done:
            raise RuntimeError("Step called on finished environment")

        if self.board[action] != 0:
            # illegal move — treat as hitting wall: heavy penalty and end episode
            self.done = True
            self.winner = 2 if player == 1 else 1
            reward = -1 if player == 1 else +1
            return self._get_state(), reward, self.done, {"illegal": True}

        self.board[action] = player

        # check winner/draw
        self.winner = self._check_winner()
        if self.winner is not None:
            self.done = True
            if self.winner == 1:
                return self._get_state(), +1, True, {}
            elif self.winner == 2:
                return self._get_state(), -1, True, {}
        elif all(cell != 0 for cell in self.board):
            # draw
            self.done = True
            return self._get_state(), 0, True, {}
        else:
            return self._get_state(), 0, False, {}

    def _check_winner(self):
        b = self.board
        lines = [
            (0,1,2),(3,4,5),(6,7,8),  # rows
            (0,3,6),(1,4,7),(2,5,8),  # cols
            (0,4,8),(2,4,6)           # diagonals
        ]
        for (i,j,k) in lines:
            if b[i] == b[j] == b[k] != 0:
                return b[i]
        return None

    def render(self):
        ch = {0: '.', 1: 'X', 2: 'O'}
        for r in range(3):
            print(' '.join(ch[self.board[3*r + c]] for c in range(3)))
        print()

# -------------------------
# Q-Learning Agent
# -------------------------
class QAgent:
    def __init__(self, alpha=0.5, gamma=0.9):
        # Q: dict mapping state -> [4..9] action-values; we'll store for all 9 actions (unused ones remain 0)
        self.Q = defaultdict(lambda: [0.0]*9)
        self.alpha = alpha
        self.gamma = gamma

    def get_action(self, state, available_actions, epsilon=0.1):
        # epsilon-greedy
        if random.random() < epsilon:
            return random.choice(available_actions)
        q_values = self.Q[state]
        # pick best among available
        best_val = max((q_values[a], a) for a in available_actions)[0]
        best_actions = [a for a in available_actions if q_values[a] == best_val]
        return random.choice(best_actions)

    def update(self, state, action, reward, next_state, next_available_actions, done):
        """
        Q(s,a) += alpha * (reward + gamma * max_a' Q(s',a') - Q(s,a))
        """
        q_sa = self.Q[state][action]
        if done:
            target = reward
        else:
            next_best = max(self.Q[next_state][a] for a in next_available_actions) if next_available_actions else 0
            target = reward + self.gamma * next_best
        self.Q[state][action] = q_sa + self.alpha * (target - q_sa)

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(dict(self.Q), f)

    def load(self, filename):
        with open(filename, 'rb') as f:
            data = pickle.load(f)
        self.Q = defaultdict(lambda: [0.0]*9, data)

# -------------------------
# Training Loop
# -------------------------
def train(agent, env, episodes=20000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9995, print_every=2000):
    epsilon = epsilon_start
    stats = {"win":0, "loss":0, "draw":0}
    for ep in range(1, episodes+1):
        state = env.reset()
        # randomly choose who starts: 1 agent, 2 opponent
        turn = random.choice([1,2])
        done = False

        # We'll keep track of the last agent state/action to update if opponent immediately wins
        last_agent = None  # tuple(state, action)

        while not done:
            if turn == 1:
                # Agent's turn
                avail = env.available_actions()
                action = agent.get_action(state, avail, epsilon)
                next_state, reward, done, info = env.step(action, player=1)
                if done:
                    # terminal after agent move (win or draw)
                    agent.update(state, action, reward, next_state, [], done=True)
                    if reward == 1:
                        stats["win"] += 1
                    elif reward == 0:
                        stats["draw"] += 1
                    else:
                        stats["loss"] += 1
                    break
                else:
                    # not done yet — opponent will move; save last agent move
                    last_agent = (state, action)
                    # update with reward 0 and lookahead later (we update after opponent move when next_state known)
                    # but do a partial update with 0 reward now (optional)
                    next_avail = env.available_actions()
                    agent.update(state, action, 0, next_state, next_avail, done=False)
                    state = next_state
                    turn = 2
            else:
                # Opponent (random) turn
                avail_op = env.available_actions()
                opp_action = random.choice(avail_op)
                next_state, reward, done, info = env.step(opp_action, player=2)
                if done:
                    # If opponent wins, that's a loss for agent. Assign -1 reward to last agent move.
                    if last_agent is not None:
                        s_last, a_last = last_agent
                        agent.update(s_last, a_last, -1, next_state, [], done=True)
                    stats["loss"] += 1
                    break
                else:
                    # continue; it's agent's turn now
                    state = next_state
                    turn = 1

        # decay epsilon
        epsilon = max(epsilon_end, epsilon * epsilon_decay)

        if ep % print_every == 0:
            total = stats["win"] + stats["loss"] + stats["draw"]
            if total == 0: total = 1
            print(f"Episode {ep}/{episodes}  eps={epsilon:.4f}  Wins={stats['win']} Losses={stats['loss']} Draws={stats['draw']}")
            # reset stats display counters (keep cumulative or reset for interval)
            stats = {"win":0, "loss":0, "draw":0}

    return agent

# -------------------------
# Test / Evaluate
# -------------------------
def evaluate(agent, env, episodes=1000, verbose=False):
    wins = draws = losses = 0
    for _ in range(episodes):
        state = env.reset()
        turn = random.choice([1,2])
        done = False
        while not done:
            if turn == 1:
                avail = env.available_actions()
                action = agent.get_action(state, avail, epsilon=0.0)  # greedy
                next_state, reward, done, info = env.step(action, player=1)
                state = next_state
                if done:
                    if reward == 1: wins += 1
                    elif reward == 0: draws += 1
                    else: losses += 1
                    break
                turn = 2
            else:
                # random opponent
                avail_op = env.available_actions()
                opp_action = random.choice(avail_op)
                next_state, reward, done, info = env.step(opp_action, player=2)
                state = next_state
                if done:
                    if reward == -1: losses += 1  # opponent win => agent loss
                    elif reward == 0: draws += 1
                    break
                turn = 1
        if verbose:
            env.render()
    total = wins + draws + losses
    return {"wins": wins, "draws": draws, "losses": losses, "win_rate": wins/total, "draw_rate": draws/total, "loss_rate": losses/total}

# -------------------------
# Play single interactive game vs human (optional)
# -------------------------
def play_against_agent(agent):
    env = TicTacToeEnv()
    state = env.reset()
    human_player = None
    while human_player not in ('X','O'):
        human_player = input("Choose X (goes first) or O (goes second): ").strip().upper()

    human = 1 if human_player == 'X' else 2
    agent_player = 2 if human == 1 else 1

    print("Game start. Board positions are 0..8 as follows:")
    print("0 1 2")
    print("3 4 5")
    print("6 7 8")
    env.render()

    turn = 1  # X always starts
    done = False
    while not done:
        if turn == human:
            while True:
                try:
                    a = int(input("Your move (0..8): "))
                    if a in env.available_actions():
                        break
                    else:
                        print("Invalid move.")
                except:
                    print("Enter an integer 0..8.")
            _, reward, done, _ = env.step(a, player=human)
            env.render()
            if done:
                if reward == 1:
                    print("X wins!") if human==1 else print("O wins!")
                elif reward == -1:
                    print("X wins!") if human==1 else print("O wins!")
                else:
                    print("Draw!")
                break
            turn = agent_player
        else:
            # agent move (greedy)
            s = env._get_state()
            a = agent.get_action(s, env.available_actions(), epsilon=0.0)
            print(f"Agent places at {a}")
            _, reward, done, _ = env.step(a, player=agent_player)
            env.render()
            if done:
                if reward == 1:
                    print("Agent wins!" if agent_player==1 else "Agent wins!")
                elif reward == -1:
                    print("Agent loses!" if agent_player==1 else "Agent loses!")
                else:
                    print("Draw!")
                break
            turn = human

# -------------------------
# Main run: train and evaluate
# -------------------------
if __name__ == "__main__":
    env = TicTacToeEnv()
    agent = QAgent(alpha=0.6, gamma=0.95)

    print("Training agent (this may take a few seconds)...")
    agent = train(agent, env, episodes=20000, epsilon_start=1.0, epsilon_end=0.05, epsilon_decay=0.9996, print_every=5000)

    print("\nEvaluating agent vs random opponent...")
    results = evaluate(agent, env, episodes=2000, verbose=False)
    print("Evaluation results:", results)

    # Save Q table
    agent.save("tictactoe_q.pkl")
    print("Q-table saved to tictactoe_q.pkl")

    # Optional: interactive play
    ans = input("Play against trained agent? (y/n): ").strip().lower()
    if ans == 'y':
        # load agent to ensure we use saved q
        agent2 = QAgent()
        agent2.load("tictactoe_q.pkl")
        play_against_agent(agent2)
