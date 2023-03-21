import random

student_name = "Jingjing Bai"


# 1. Q-Learning
class QLearningAgent:
    """Implement Q Reinforcement Learning Agent using Q-table."""

    def __init__(self, game, discount, learning_rate, explore_prob):
        """Store any needed parameters into the agent object.
        Initialize Q-table.
        """
        self.game = game
        self.discount = discount
        self.alpha = learning_rate
        self.epsilon = explore_prob
        self.qvalues = {}

    def get_q_value(self, state, action):
        """Retrieve Q-value from Q-table.
        For an never seen (s,a) pair, the Q-value is by default 0.
        """
        if (state,action) in self.qvalues:
            return self.qvalues[(state,action)]
        else:
            return 0.0

    def set_q_value(self, state, action, value):
        self.qvalues[(state, action)] = value

    def get_value(self, state):
        """Compute state value from Q-values using Bellman Equation.
        V(s) = max_a Q(s,a)
        """
        qvalues = [self.get_q_value(state, action) for action in self.game.get_actions(state)]
        if not len(qvalues): return 0.0
        return max(qvalues)

    def get_best_policy(self, state):
        """Compute the best action to take
        in the state using Policy Extraction.
        π(s) = argmax_a Q(s,a)

        If there are ties, return a random one for better performance.
        Hint: use random.choice().
        """
        import random
        best_value = self.get_value(state)
        best_actions = [action for action in self.game.get_actions(state) \
                        if self.get_q_value(state, action) == best_value]
    
        if not len(best_actions): return None
        else: 
            #print("before", best_actions)
            r = random.choice(best_actions)
            #print(state, r)
            return r 

    def update(self, state, action, next_state, reward):
        """Update Q-values using running average.
        Q(s,a) = (1 - α) Q(s,a) + α (R + γ V(s'))
        Where α is the learning rate, and γ is the discount.

        Note: You should not call this function in your code.
        """
        qvalue = self.get_q_value(state, action)
        next_value = self.get_value(next_state)
        new_value = (1-self.alpha) * qvalue + self.alpha * (reward + self.discount * next_value)
        self.set_q_value(state, action, new_value)

    # 2. Epsilon Greedy
    def get_action(self, state):
        """Compute the action to take for the agent, incorporating exploration.
        That is, with probability ε, act randomly.
        Otherwise, act according to the best policy.

        Hint: use random.random() < ε to check if exploration is needed.
        """
        legal_actions = self.game.get_actions(state)
        action = None
        #print("get_action for", state, "legal_actions", legal_actions)
        if random.random() < self.epsilon:
            action = random.choice(list(legal_actions))
        else:
            action = self.get_best_policy(state)
        #print("action is ", action)
        return action


# 3. Bridge Crossing Revisited
def question3():
    epsilon = ...
    learning_rate = ...
    return 'NOT POSSIBLE'
    # If not possible, return 'NOT POSSIBLE'


# 5. Approximate Q-Learning
class ApproximateQAgent(QLearningAgent):
    """Implement Approximate Q Learning Agent using weights."""

    def __init__(self, *args, extractor):
        """Initialize parameters and store the feature extractor.
        Initialize weights table."""

        super().__init__(*args)
        ...  # TODO

    def get_weight(self, feature):
        """Get weight of a feature.
        Never seen feature should have a weight of 0.
        """
        return 0  # TODO

    def get_q_value(self, state, action):
        """Compute Q value based on the dot product
        of feature components and weights.
        Q(s,a) = w_1 * f_1(s,a) + w_2 * f_2(s,a) + ... + w_n * f_n(s,a)
        """
        return 0  # TODO

    def update(self, state, action, next_state, reward):
        """Update weights using least-squares approximation.
        Δ = R + γ V(s') - Q(s,a)
        Then update weights: w_i = w_i + α * Δ * f_i(s, a)
        """
        ...  # TODO


# 6. Feedback
# Just an approximation is fine.
feedback_question_1 = 0

feedback_question_2 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""

feedback_question_3 = """
Type your response here.
Your response may span multiple lines.
Do not include these instructions in your response.
"""
