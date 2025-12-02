import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from collections import deque
import random
from sklearn.model_selection import train_test_split

# Convert numpy arrays to torch tensors
def convert_to_torch_tensors(X, y):
    X = torch.from_numpy(X).float()
    y = torch.from_numpy(y).long()
    return X, y

class SimpleClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleClassifier, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)
        return x

    def predict(self, x):
        return self.forward(x).argmax(dim=1)

    def predict_proba(self, x):
        return torch.softmax(self.forward(x), dim=1)

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path, device=None, strict=True):
        try:
            if device is None:
                state_dict = torch.load(path, map_location='cpu')
            else:
                # Convert device to string for map_location
                map_location = str(device) if isinstance(device, torch.device) else device
                state_dict = torch.load(path, map_location=map_location)
            
            self.load_state_dict(state_dict, strict=strict)
            return True
        except (RuntimeError, KeyError) as e:
            print(f"Warning: Failed to load model from {path}: {e}")
            return False

def train_classifier(classifier, X, y, device, epochs=1000, batch_size=256, lr=1e-4):
    classifier.to(device)
    optimizer = optim.Adam(classifier.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()
    best_test_acc = 0.0
    best_model_state = None
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, y_train = convert_to_torch_tensors(X_train, y_train)
    X_test, y_test = convert_to_torch_tensors(X_test, y_test)
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
    optimizerLR = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    for epoch in range(epochs):
        classifier.train()
        epoch_loss = 0.0
        for x_batch, y_batch in train_loader:
            x_batch = x_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = classifier(x_batch)
            loss = criterion(logits, y_batch.squeeze())
            loss.backward()
            torch.nn.utils.clip_grad_norm_(classifier.parameters(), max_norm=1.0)
            epoch_loss += loss.item()
            optimizer.step()
        classifier.eval()
        with torch.no_grad():
            test_logits = classifier(X_test.to(device))
            test_preds = test_logits.argmax(dim=1)
            test_acc = accuracy_score(y_test.cpu().numpy(), test_preds.cpu().numpy())
            optimizerLR.step(test_acc)
            if epoch % 100 == 0:
                print(f"Epoch {epoch+1}/{epochs} | Loss: {epoch_loss/len(train_loader):.4f} | Test Acc: {test_acc:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}")
            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_model_state = classifier.state_dict().copy()

class PolicyNet(nn.Module):
    def __init__(self, agent_id, input_size, hidden_size, output_size, device):
        super(PolicyNet, self).__init__()
        self.agent_id = agent_id
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(0.3)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.device = device
        self.to(self.device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.dropout(x)
        x = self.fc3(x)  # Remove ReLU before tanh - tanh already bounds output to [-1, 1]
        x = torch.tanh(x)
        return x

class MultiAgentPolicyNet(nn.Module):
    def __init__(self, num_agents, input_size, hidden_size, output_size, device):
        super(MultiAgentPolicyNet, self).__init__()
        self.policy_nets = nn.ModuleList([PolicyNet(agent_id, input_size, hidden_size, output_size, device) for agent_id in range(num_agents)])
        self.device = device
        self.to(self.device)

    def forward(self, x):
        return torch.stack([policy_net.forward(x) for policy_net in self.policy_nets], dim=1)

class CentralizedCritic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device):
        super(CentralizedCritic, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)
        self.batch_norm1 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.batch_norm2 = nn.BatchNorm1d(hidden_size, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
        self.device = device
        self.to(self.device)
        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.batch_norm1(x)
        x = self.relu(self.fc2(x))
        x = self.batch_norm2(x)
        x = self.tanh(self.fc3(x))
        return x

class globalBuffer():
    def __init__(self, run_config):
        self.run_config = run_config
        self.buffer = deque(maxlen=run_config["buffer_size"])
        self.sample_id = 0

    def add(self, agent_id, obs, action, reward, next_obs, done, info):
        transition = {
            "agent_id": agent_id,
            "obs": obs,
            "action": action,
            "reward": reward,
            "next_obs": next_obs,
            "done": done,
            "info": info
        }
        self.buffer.append(transition)
        self.sample_id += 1
        return self.sample_id

    def current_buffer_size(self):
        return len(self.buffer)

    def is_buffer_full(self):
        return len(self.buffer) == self.buffer.maxlen

    def reset_buffer(self):
        self.buffer.clear()
        self.sample_id = 0

    def random_sample(self, batch_size):
        """Sample a batch of transitions from the buffer."""
        if len(self.buffer) == 0:
            return []
        if batch_size > len(self.buffer):
            batch_size = len(self.buffer)
        return random.sample(self.buffer, batch_size)
        

class AnchorEnv():
    def __init__(self, obs_space, action_space, agent_id, run_config):
        super(AnchorEnv, self).__init__()
        self.obs_space = obs_space
        self.action_space = action_space
        self.agent_id = agent_id
        self.run_config = run_config
        self.current_obs = None
        self.device = run_config.get("device", torch.device("cpu"))

    def get_reward_function(self, obs, action):
        pass

    def get_next_obs(self, current_obs, action):
        return self.obs_space.sample()

    def step(self, action):
        if self.current_obs is None:
            self.current_obs = self.obs_space.sample()
        next_obs = self.get_next_obs(self.current_obs, action)
        reward = self.get_reward_function(next_obs, action)
        done = False
        truncated = False
        info = {}
        self.current_obs = next_obs
        return next_obs, reward, done, truncated, info

    def reset(self, seed=None):
        if seed is not None:
            self.seed(seed)
        self.current_obs = self.obs_space.sample()
        return self.current_obs

    def seed(self, seed):
        if hasattr(self.obs_space, 'seed'):
            self.obs_space.seed(seed)

    def close(self):
        self.current_obs = None

    def render(self):
        return None

    def get_obs_space(self):
        return self.obs_space

    def get_action_space(self):
        return self.action_space

    def generate_rule_from_anchors(self, feature_names, anchors):
        rule = ""
        for i in range(len(feature_names)):
            rule += f"if {feature_names[i]} <= x <= {anchors[i]} then y = {anchors[i]}"
            if i < len(feature_names) - 1:
                rule += " and "
        return rule

class MultiAgentEnvironment:
    def __init__(self, classifier, centralized_critic, policy_nets, n_agents, obs_space, action_space, run_config):
        self.classifier = classifier
        self.centralized_critic = centralized_critic
        self.policy_nets = policy_nets
        self.n_agents = n_agents
        self.anchor_envs = [AnchorEnv(obs_space[agent_id], action_space[agent_id], agent_id, run_config) for agent_id in range(n_agents)]
        self.run_config = run_config
        self.device = run_config.get("device", torch.device("cpu"))

    def seed(self, seed):
        for anchor_env in self.anchor_envs:
            anchor_env.seed(seed)

    def reset(self, agent_id, seed=None):
        return self.anchor_envs[agent_id].reset(seed=seed)

    def step(self, agent_id, action):
        return self.anchor_envs[agent_id].step(action)

    def close(self):
        for anchor_env in self.anchor_envs:
            anchor_env.close()

    def render(self, agent_id):
        return self.anchor_envs[agent_id].render()

    def get_observation_space(self, agent_id):
        return self.anchor_envs[agent_id].get_obs_space()

    def get_action_space(self, agent_id):
        return self.anchor_envs[agent_id].get_action_space()


class CentralizedTrainer():
    def __init__(self, multiagent_env, n_agents, obs_space, action_space, run_config):
        self.multiagent_env = multiagent_env
        self.centralized_critic = multiagent_env.centralized_critic
        self.policy_nets = multiagent_env.policy_nets
        self.n_agents = n_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.run_config = run_config
        self.buffer = globalBuffer(run_config)
        self.device = run_config.get("device", torch.device("cpu"))
        self.learning_starts = run_config.get("learning_starts", 1000)
        self.train_frequency = run_config.get("train_frequency", 1)
        self.batch_size = run_config.get("batch_size", 256)

    def train(self):
        total_steps = 0
        training_stats = {
            "episodes_completed": 0,
            "samples_collected": 0,
            "training_updates": 0,
            "buffer_size": []
        }
        
        print(f"Starting training with warm-up phase: {self.learning_starts} samples")
        print("="*80)
        
        for episode in range(self.run_config["num_episodes"]):
            print(f"Episode {episode + 1}/{self.run_config['num_episodes']}")
            for agent_id in range(self.n_agents):
                print(f"Agent id: {agent_id}")

        return {
            "buffer": self.buffer,
            "stats": training_stats
        }
    
    def _train_step(self):
        """
        Perform one training step on a batch from the replay buffer.
        This updates both the centralized critic and policy networks.
        """
        if self.buffer.current_buffer_size() < self.batch_size:
            return
        
        # Sample a batch from the buffer
        batch = self.buffer.random_sample(self.batch_size)
        if len(batch) == 0:
            return
        
        # TODO: Implement actual training logic here
        # 1. Extract batch data (obs, actions, rewards, next_obs, dones)
        # 2. Compute centralized value estimates using centralized_critic
        # 3. Compute advantages (rewards + gamma * V(next_obs) - V(obs))
        # 4. Update policy networks using policy gradient
        # 5. Update centralized critic using TD error
        
        # Placeholder - you'll implement the actual training
        pass

class DecentralizedExecutor():
    def __init__(self, multiagent_env, policy_nets, n_agents, obs_space, action_space, run_config):
        self.multiagent_env = multiagent_env
        self.policy_nets = policy_nets
        self.n_agents = n_agents
        self.obs_space = obs_space
        self.action_space = action_space
        self.run_config = run_config

    def execute(self):
        pass