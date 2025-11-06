import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, TensorDataset
import json


class SimpleClassifier(nn.Module):
    def __init__(self, input_dim: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PolicyNet(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.net(x)
        return logits


class ValueNet(nn.Module):
    def __init__(self, state_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).view(-1)


# Import environment classes from dyn_anchor_PPO
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from dyn_anchor_PPO import AnchorEnv, DiscreteAnchorEnv, compute_quantile_bins, discretize_by_edges, compute_bin_representatives, select_device


def train_classifier_fully(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    num_classes: int,
    device: torch.device,
    epochs: int = 100,
    batch_size: int = 256,
    lr: float = 1e-3,
    patience: int = 10,
    verbose: bool = True,
):
    """
    Train classifier to completion before RL training.
    
    Args:
        X_train: Training features (already standardized)
        y_train: Training labels
        X_test: Test features (already standardized)
        y_test: Test labels
        num_classes: Number of classes
        device: Torch device
        epochs: Maximum epochs
        batch_size: Batch size
        lr: Learning rate
        patience: Early stopping patience
        verbose: Whether to print progress
    
    Returns:
        Trained classifier and training history
    """
    classifier = SimpleClassifier(X_train.shape[1], num_classes).to(device)
    clf_opt = optim.Adam(classifier.parameters(), lr=lr)
    ce = nn.CrossEntropyLoss()
    
    train_losses = []
    train_accs = []
    test_accs = []
    best_test_acc = 0.0
    best_epoch = 0
    patience_counter = 0
    
    dataset = TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(y_train).long())
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    for epoch in range(1, epochs + 1):
        classifier.train()
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0
        
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            clf_opt.zero_grad()
            logits = classifier(xb)
            loss = ce(logits, yb)
            loss.backward()
            clf_opt.step()
            
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                correct = (preds == yb).sum().item()
                epoch_correct += correct
                epoch_count += yb.size(0)
                epoch_loss_sum += loss.item() * yb.size(0)
        
        train_loss = epoch_loss_sum / max(1, epoch_count)
        train_acc = epoch_correct / max(1, epoch_count)
        
        # Evaluate on test set
        classifier.eval()
        with torch.no_grad():
            inputs_test = torch.from_numpy(X_test).float().to(device)
            logits_test = classifier(inputs_test)
            preds_test = logits_test.argmax(dim=1).cpu().numpy()
            test_acc = accuracy_score(y_test, preds_test)
        
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        test_accs.append(test_acc)
        
        if test_acc > best_test_acc:
            best_test_acc = test_acc
            best_epoch = epoch
            patience_counter = 0
        else:
            patience_counter += 1
        
        if verbose and epoch % 10 == 0:
            print(f"[clf] epoch {epoch}/{epochs} | loss={train_loss:.4f} | train_acc={train_acc:.3f} | test_acc={test_acc:.3f} | patience={patience_counter}/{patience}")
        
        if patience_counter >= patience and epoch >= 50:
            if verbose:
                print(f"[clf] Early stopping at epoch {epoch} (best test_acc={best_test_acc:.3f} at epoch {best_epoch})")
            break
    
    if verbose:
        print(f"[clf] Training complete: best_test_acc={best_test_acc:.3f} at epoch {best_epoch}")
    
    return classifier, {
        'train_losses': train_losses,
        'train_accs': train_accs,
        'test_accs': test_accs,
        'best_epoch': best_epoch,
        'best_test_acc': best_test_acc,
    }


def train_rl_policy_posthoc(
    envs: dict,
    policy: PolicyNet,
    value_fn: ValueNet,
    device: torch.device,
    episodes: int = 50,
    steps_per_episode: int = 40,
    gamma: float = 0.99,
    ppo_epochs: int = 4,
    clip_epsilon: float = 0.2,
    entropy_coef: float = 0.02,
    value_coef: float = 0.5,
    batch_size: int | None = None,
    lr: float = 3e-4,
    verbose: bool = True,
):
    """
    Train RL policy post-hoc on frozen trained classifier.
    
    Args:
        envs: Dictionary of environments (one per class)
        policy: Policy network
        value_fn: Value network
        device: Torch device
        episodes: Number of RL episodes
        steps_per_episode: Steps per episode
        gamma: Discount factor
        ppo_epochs: PPO update epochs
        clip_epsilon: PPO clipping parameter
        entropy_coef: Entropy regularization coefficient
        value_coef: Value loss coefficient
        batch_size: Batch size (None = use all experiences)
        lr: Learning rate
        verbose: Whether to print progress
    
    Returns:
        Training history
    """
    policy_opt = optim.Adam(list(policy.parameters()) + list(value_fn.parameters()), lr=lr)
    
    episode_rewards = []
    per_class_prec_cov = {c: [] for c in envs.keys()}
    per_class_box_history = {c: [] for c in envs.keys()}
    
    for ep in range(episodes):
        # Collect experiences from all classes
        all_states = []
        all_actions = []
        all_old_log_probs = []
        all_rewards = []
        all_values = []
        all_dones = []
        total_return = 0.0
        
        for cls, env in envs.items():
            state = env.reset()
            class_states = []
            class_actions = []
            class_old_log_probs = []
            class_rewards = []
            class_values = []
            class_dones = []
            
            for t in range(steps_per_episode):
                s = torch.from_numpy(state).float().to(device)
                with torch.no_grad():
                    logits = policy(s)
                    probs_pi = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs_pi)
                    action = dist.sample()
                    log_prob = dist.log_prob(action)
                    value = value_fn(s)
                
                next_state, reward, done, info = env.step(int(action.item()))
                
                class_states.append(state.copy())
                class_actions.append(int(action.item()))
                class_old_log_probs.append(log_prob.item())
                class_rewards.append(float(reward))
                class_values.append(value.item())
                class_dones.append(bool(done))
                
                state = next_state
                if done:
                    break
            
            all_states.extend(class_states)
            all_actions.extend(class_actions)
            all_old_log_probs.extend(class_old_log_probs)
            all_rewards.extend(class_rewards)
            all_values.extend(class_values)
            all_dones.extend(class_dones)
            
            total_return += sum(class_rewards) if class_rewards else 0.0
            
            # Log metrics
            last_info = info if 'info' in locals() else {}
            per_class_prec_cov[cls].append({
                'precision': last_info.get('precision', 0.0),
                'hard_precision': last_info.get('hard_precision', 0.0),
                'coverage': last_info.get('coverage', 0.0),
            })
            per_class_box_history[cls].append((env.lower.copy(), env.upper.copy()))
        
        # PPO: Compute returns and advantages
        if len(all_rewards) > 0:
            returns = []
            R = 0.0
            for i in reversed(range(len(all_rewards))):
                if all_dones[i]:
                    R = 0.0
                R = all_rewards[i] + gamma * R
                returns.insert(0, R)
            
            returns_t = torch.tensor(returns, dtype=torch.float32, device=device)
            values_t = torch.tensor(all_values, dtype=torch.float32, device=device)
            advantages = returns_t - values_t
            
            # Standardize advantages
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False) + 1e-8
            advantages = (advantages - adv_mean) / adv_std
            
            # Convert to tensors
            states_t = torch.tensor(np.array(all_states), dtype=torch.float32, device=device)
            actions_t = torch.tensor(all_actions, dtype=torch.long, device=device)
            old_log_probs_t = torch.tensor(all_old_log_probs, dtype=torch.float32, device=device)
            old_values_t = values_t.clone().detach()
            
            # PPO: Multiple update epochs
            for ppo_epoch in range(ppo_epochs):
                policy_opt.zero_grad()
                
                # Create batches if batch_size is specified
                if batch_size is not None and batch_size < len(states_t):
                    indices = torch.randperm(len(states_t), device=device)
                    total_loss = 0.0
                    n_batches = 0
                    
                    for batch_start in range(0, len(states_t), batch_size):
                        batch_end = min(batch_start + batch_size, len(states_t))
                        batch_indices = indices[batch_start:batch_end]
                        
                        batch_states = states_t[batch_indices]
                        batch_actions = actions_t[batch_indices]
                        batch_old_log_probs = old_log_probs_t[batch_indices]
                        batch_advantages = advantages[batch_indices]
                        batch_returns = returns_t[batch_indices]
                        batch_old_values = old_values_t[batch_indices]
                        
                        # Re-evaluate policy and value
                        logits = policy(batch_states)
                        probs_pi = torch.softmax(logits, dim=-1)
                        dist = torch.distributions.Categorical(probs_pi)
                        new_log_probs = dist.log_prob(batch_actions)
                        new_values = value_fn(batch_states)
                        entropy = dist.entropy().mean()
                        
                        # Importance sampling ratio
                        ratio = torch.exp(new_log_probs - batch_old_log_probs)
                        
                        # Clipped policy loss
                        surr1 = ratio * batch_advantages
                        surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * batch_advantages
                        policy_loss = -torch.min(surr1, surr2).mean()
                        
                        # Value loss
                        value_loss = 0.5 * (new_values - batch_returns).pow(2).mean()
                        
                        # Entropy bonus
                        entropy_term = -entropy_coef * entropy
                        
                        # Total loss
                        loss = policy_loss + value_coef * value_loss + entropy_term
                        loss.backward()
                        
                        total_loss += loss.item()
                        n_batches += 1
                    
                    # Gradient clipping and update
                    torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), max_norm=0.5)
                    policy_opt.step()
                    
                else:
                    # Use all experiences as one batch
                    logits = policy(states_t)
                    probs_pi = torch.softmax(logits, dim=-1)
                    dist = torch.distributions.Categorical(probs_pi)
                    new_log_probs = dist.log_prob(actions_t)
                    new_values = value_fn(states_t)
                    entropy = dist.entropy().mean()
                    
                    # Importance sampling ratio
                    ratio = torch.exp(new_log_probs - old_log_probs_t)
                    
                    # Clipped policy loss
                    surr1 = ratio * advantages
                    surr2 = torch.clamp(ratio, 1.0 - clip_epsilon, 1.0 + clip_epsilon) * advantages
                    policy_loss = -torch.min(surr1, surr2).mean()
                    
                    # Value loss
                    value_loss = 0.5 * (new_values - returns_t).pow(2).mean()
                    
                    # Entropy bonus
                    entropy_term = -entropy_coef * entropy
                    
                    # Total loss
                    loss = policy_loss + value_coef * value_loss + entropy_term
                    loss.backward()
                    
                    # Gradient clipping and update
                    torch.nn.utils.clip_grad_norm_(list(policy.parameters()) + list(value_fn.parameters()), max_norm=0.5)
                    policy_opt.step()
        
        episode_rewards.append(total_return)
        
        if verbose and (ep + 1) % 10 == 0:
            print(f"[RL] episode {ep+1}/{episodes} | return={total_return:.3f}")
    
    return {
        'episode_rewards': episode_rewards,
        'per_class_precision_coverage': per_class_prec_cov,
        'per_class_box_history': per_class_box_history,
    }


def train_posthoc_dynamic_anchors(
    dataset: str = "covtype",
    seed: int = 42,
    device_preference: str = "auto",
    classifier_epochs: int = 100,
    classifier_batch_size: int = 256,
    classifier_lr: float = 1e-3,
    classifier_patience: int = 10,
    rl_episodes: int = None,  # Will use dataset-specific defaults
    rl_steps_per_episode: int = None,  # Will use dataset-specific defaults
    rl_ppo_epochs: int = 4,
    rl_clip_epsilon: float = 0.2,
    rl_entropy_coef: float = None,  # Will use dataset-specific defaults
    rl_value_coef: float = 0.5,
    rl_batch_size: int | None = None,
    rl_lr: float = 3e-4,
    use_perturbation: bool = None,  # Will use dataset-specific defaults
    perturbation_mode: str = None,  # Will use dataset-specific defaults
    n_perturb: int = None,  # Will use dataset-specific defaults
    use_discretization: bool = True,
    disc_perc: list[int] | None = None,
    show_plots: bool = False,
    num_greedy_rollouts: int = 20,
    max_features_in_rule: int = 5,
):
    """
    Train dynamic anchors in post-hoc fashion:
    1. Train classifier fully to completion
    2. Train RL policy post-hoc on frozen classifier
    3. Freeze policy and evaluate
    
    Args:
        dataset: Dataset name ("breast_cancer", "synthetic", "covtype")
        seed: Random seed
        device_preference: Device selection
        classifier_epochs: Max classifier epochs
        classifier_batch_size: Classifier batch size
        classifier_lr: Classifier learning rate
        classifier_patience: Early stopping patience
        rl_episodes: Number of RL episodes
        rl_steps_per_episode: RL steps per episode
        rl_ppo_epochs: PPO update epochs
        rl_clip_epsilon: PPO clipping parameter
        rl_entropy_coef: RL entropy coefficient
        rl_value_coef: RL value coefficient
        rl_batch_size: RL batch size (None = all experiences)
        rl_lr: RL learning rate
        use_perturbation: Use perturbation sampling
        perturbation_mode: Perturbation mode
        n_perturb: Number of perturbations
        use_discretization: Use discretization
        disc_perc: Discretization percentiles
        show_plots: Show plots
        num_greedy_rollouts: Number of greedy rollouts per class
        max_features_in_rule: Max features in rule
    
    Returns:
        Results dictionary
    """
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)
    
    # Dataset loading
    if dataset == "breast_cancer":
        ds = load_breast_cancer()
        X = ds.data.astype(np.float32)
        y = ds.target.astype(int)
        feature_names = list(ds.feature_names)
    elif dataset == "synthetic":
        from sklearn.datasets import make_classification
        X, y = make_classification(n_samples=2000, n_features=12, n_informative=8, n_classes=2, random_state=seed)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    elif dataset == "covtype":
        X, y = fetch_covtype(return_X_y=True, as_frame=False)
        X = X.astype(np.float32)
        y = y.astype(int)
        feature_names = [f"f{i}" for i in range(X.shape[1])]
    else:
        raise ValueError(f"Unknown dataset '{dataset}'.")
    
    # Normalize class labels
    unique_classes = np.unique(y)
    if dataset == "breast_cancer":
        class_names = [str(load_breast_cancer().target_names[c]) for c in unique_classes]
    else:
        class_names = [f"class_{int(c)}" for c in unique_classes]
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_to_idx[c] for c in y], dtype=int)
    num_classes = int(len(unique_classes))
    target_classes = tuple(range(num_classes))
    
    # Dataset-specific presets
    presets = {
        "breast_cancer": {
            "rl_episodes": 25,
            "rl_steps_per_episode": 40,
            "rl_entropy_coef": 0.02,
            "use_perturbation": False,
            "perturbation_mode": "bootstrap",
            "n_perturb": 1024,
            "step_fracs": (0.01, 0.02, 0.04),
            "min_width": 0.05,
            "precision_target": 0.95,
            "coverage_target": 0.05,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [25, 50, 75],
        },
        "synthetic": {
            "rl_episodes": 30,
            "rl_steps_per_episode": 50,
            "rl_entropy_coef": 0.02,
            "use_perturbation": True,
            "perturbation_mode": "uniform",
            "n_perturb": 2048,
            "step_fracs": (0.005, 0.01, 0.02),
            "min_width": 0.04,
            "precision_target": 0.95,
            "coverage_target": 0.04,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [20, 40, 60, 80],
        },
        "covtype": {
            "rl_episodes": 60,
            "rl_steps_per_episode": 90,
            "rl_entropy_coef": 0.015,
            "use_perturbation": True,
            "perturbation_mode": "uniform",
            "n_perturb": 8192,
            "step_fracs": (0.003, 0.006, 0.012),
            "min_width": 0.02,
            "precision_target": 0.95,
            "coverage_target": 0.02,
            "precision_blend_lambda": 0.5,
            "drift_penalty_weight": 0.05,
            "js_penalty_weight": 0.05,
            "disc_perc": [10, 25, 50, 75, 90],
        },
    }
    p = presets[dataset]
    
    # Resolve parameters (use argument if provided, else use preset defaults)
    rl_episodes = int(rl_episodes if rl_episodes is not None else p["rl_episodes"])
    rl_steps_per_episode = int(rl_steps_per_episode if rl_steps_per_episode is not None else p["rl_steps_per_episode"])
    rl_entropy_coef = float(rl_entropy_coef if rl_entropy_coef is not None else p["rl_entropy_coef"])
    use_perturbation = bool(use_perturbation if use_perturbation is not None else p["use_perturbation"])
    perturbation_mode = str(perturbation_mode if perturbation_mode is not None else p["perturbation_mode"])
    n_perturb = int(n_perturb if n_perturb is not None else p["n_perturb"])
    disc_perc = disc_perc if disc_perc is not None else p["disc_perc"]
    
    print("*****************")
    print("")
    print("Run configuration (Post-hoc)")
    print("")
    print("*****************")
    print(f"[data] classes ({num_classes}): {class_names} | feature_names ({len(feature_names)}): {feature_names}")
    print(f"[config] dataset={dataset}, classifier_epochs={classifier_epochs}, rl_episodes={rl_episodes}, rl_steps={rl_steps_per_episode}")
    print(f"[config] use_perturbation={use_perturbation}, mode={perturbation_mode}, n_perturb={n_perturb}")
    
    # Split and scale
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)
    
    # Build unit-space stats
    X_min = X_train.min(axis=0)
    X_max = X_train.max(axis=0)
    X_range = np.where((X_max - X_min) == 0, 1.0, (X_max - X_min))
    X_unit_train = (X_train - X_min) / X_range
    X_unit_test = (X_test - X_min) / X_range
    
    device = select_device(device_preference)
    print(f"[device] using {device}")
    
    # ========== STEP 1: Train classifier fully ==========
    print("\n" + "="*60)
    print("STEP 1: Training Classifier")
    print("="*60)
    classifier, clf_history = train_classifier_fully(
        X_train, y_train, X_test, y_test, num_classes, device,
        epochs=classifier_epochs,
        batch_size=classifier_batch_size,
        lr=classifier_lr,
        patience=classifier_patience,
        verbose=True,
    )
    
    # ========== STEP 2: Create environments with frozen classifier ==========
    print("\n" + "="*60)
    print("STEP 2: Creating Environments with Frozen Classifier")
    print("="*60)
    
    edges = None
    if use_discretization:
        edges = compute_quantile_bins(X_train, disc_perc)
        X_bins_train = discretize_by_edges(X_train, edges)
        bin_reps = compute_bin_representatives(X_train, X_bins_train)
        envs = {
            c: DiscreteAnchorEnv(
                X_bins=X_bins_train,
                X_std=X_train,
                y=y_train,
                feature_names=feature_names,
                classifier=classifier,
                device=device,
                bin_reps=bin_reps,
                bin_edges=edges,
                target_class=c,
                step_fracs=(1, 1, 1),
                min_width=1.0,
                alpha=0.7,
                beta=0.6,
                gamma=0.1,
                precision_target=p["precision_target"],
                coverage_target=p["coverage_target"],
                precision_blend_lambda=p["precision_blend_lambda"],
                drift_penalty_weight=p["drift_penalty_weight"],
                use_perturbation=use_perturbation,
                perturbation_mode=perturbation_mode,
                n_perturb=n_perturb,
                rng=rng,
                min_coverage_floor=0.05,
                js_penalty_weight=p["js_penalty_weight"],
                x_star_bins=None,
            ) for c in target_classes
        }
    else:
        envs = {
            c: AnchorEnv(
                X_unit_train, X_train, y_train, feature_names, classifier, device,
                target_class=c,
                step_fracs=p["step_fracs"],
                min_width=p["min_width"],
                alpha=0.7,
                beta=0.6,
                gamma=0.1,
                precision_target=p["precision_target"],
                coverage_target=p["coverage_target"],
                precision_blend_lambda=p["precision_blend_lambda"],
                drift_penalty_weight=p["drift_penalty_weight"],
                use_perturbation=p["use_perturbation"],
                perturbation_mode=perturbation_mode,
                n_perturb=n_perturb,
                X_min=X_min, X_range=X_range,
                rng=rng,
                x_star_unit=None,
                initial_window=0.1,
                js_penalty_weight=p["js_penalty_weight"],
            ) for c in target_classes
        }
    
    # ========== STEP 3: Train RL policy post-hoc ==========
    print("\n" + "="*60)
    print("STEP 3: Training RL Policy (Post-hoc)")
    print("="*60)
    
    any_env = next(iter(envs.values()))
    state_dim = 2 * any_env.n_features + 2
    action_dim = any_env.n_actions
    
    policy = PolicyNet(state_dim, action_dim).to(device)
    value_fn = ValueNet(state_dim).to(device)
    
    rl_history = train_rl_policy_posthoc(
        envs=envs,
        policy=policy,
        value_fn=value_fn,
        device=device,
        episodes=rl_episodes,
        steps_per_episode=rl_steps_per_episode,
        ppo_epochs=rl_ppo_epochs,
        clip_epsilon=rl_clip_epsilon,
        entropy_coef=rl_entropy_coef,
        value_coef=rl_value_coef,
        batch_size=rl_batch_size,
        lr=rl_lr,
        verbose=True,
    )
    
    # ========== STEP 4: Freeze policy and evaluate ==========
    print("\n" + "="*60)
    print("STEP 4: Evaluating with Frozen Policy")
    print("="*60)
    
    policy.eval()
    value_fn.eval()
    
    def greedy_rollout(env, policy, device, steps_per_episode):
        state = env.reset()
        initial_lower = env.lower.copy()
        initial_upper = env.upper.copy()
        initial_width = (initial_upper - initial_lower)
        last_info = {"precision": 0.0, "coverage": 0.0, "hard_precision": 0.0}
        
        for t in range(steps_per_episode):
            s = torch.from_numpy(state).float().to(device)
            with torch.no_grad():
                logits = policy(s)
                action = int(torch.argmax(logits, dim=-1).item())
            next_state, _, done, info = env.step(action)
            state = next_state
            last_info = info
            if done:
                break
        
        # Build rule
        lw = (env.upper - env.lower)
        tightened = np.where(lw < initial_width * 0.95)[0]
        
        if tightened.size == 0:
            rule = "any values (no tightened features)"
        else:
            tightened_sorted = np.argsort(lw[tightened])
            to_show_idx = (tightened[tightened_sorted[:max_features_in_rule]] if max_features_in_rule > 0 else tightened)
            cond_parts = []
            for i in to_show_idx:
                if hasattr(env, 'bin_edges') and isinstance(env, DiscreteAnchorEnv):
                    lbin = int(np.floor(env.lower[i]))
                    ubin = int(np.ceil(env.upper[i]))
                    edges_i = env.bin_edges[i]
                    feat_min = float(env.X_std[:, i].min())
                    feat_max = float(env.X_std[:, i].max())
                    if lbin <= 0:
                        left = feat_min
                    else:
                        left = float(edges_i[min(lbin-1, len(edges_i)-1)])
                    if ubin >= len(edges_i):
                        right = feat_max
                    else:
                        right = float(edges_i[ubin])
                    if left <= feat_min + 1e-6 and right >= feat_max - 1e-6:
                        continue
                    elif left <= feat_min + 1e-6:
                        cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                    elif right >= feat_max - 1e-6:
                        cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                    else:
                        cond_parts.append(f"{feature_names[i]} > {left:.2f}")
                        cond_parts.append(f"{feature_names[i]} <= {right:.2f}")
                else:
                    cond_parts.append(f"{feature_names[i]} ∈ [{env.lower[i]:.2f}, {env.upper[i]:.2f}]")
            rule = " and ".join(cond_parts)
        
        return last_info, rule, env.lower.copy(), env.upper.copy()
    
    # Greedy evaluation
    final_greedy = {}
    final_greedy_all = {}
    
    for cls in target_classes:
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        
        sel = rng.choice(idx_cls, size=min(num_greedy_rollouts, idx_cls.size), replace=False)
        cls_name = class_names[cls] if cls < len(class_names) else str(cls)
        print(f"[greedy] Class {cls} ({cls_name}): Running {len(sel)} greedy rollouts...")
        
        all_anchor_results = []
        for i, instance_idx in enumerate(sel):
            if (i + 1) % 5 == 0 or (i + 1) == len(sel):
                print(f"  [greedy cls={cls}] Progress: {i+1}/{len(sel)} instances", end='\r')
            
            if use_discretization:
                env = DiscreteAnchorEnv(
                    X_bins=discretize_by_edges(X_test, edges),
                    X_std=X_test,
                    y=y_test,
                    feature_names=feature_names,
                    classifier=classifier,
                    device=device,
                    bin_reps=bin_reps,
                    bin_edges=edges,
                    target_class=cls,
                    step_fracs=(1,1,1),
                    min_width=1.0,
                    alpha=0.7,
                    beta=0.6,
                    gamma=0.1,
                    precision_target=p["precision_target"],
                    coverage_target=p["coverage_target"],
                    precision_blend_lambda=p["precision_blend_lambda"],
                    drift_penalty_weight=p["drift_penalty_weight"],
                    use_perturbation=use_perturbation,
                    perturbation_mode=perturbation_mode,
                    n_perturb=n_perturb,
                    rng=np.random.default_rng(seed + instance_idx * 1000),
                    min_coverage_floor=0.05,
                    js_penalty_weight=p["js_penalty_weight"],
                    x_star_bins=None,
                )
            else:
                env = AnchorEnv(
                    X_unit_test, X_test, y_test, feature_names, classifier, device,
                    target_class=cls,
                    step_fracs=p["step_fracs"],
                    min_width=p["min_width"],
                    alpha=0.7,
                    beta=0.6,
                    gamma=0.1,
                    precision_target=p["precision_target"],
                    coverage_target=p["coverage_target"],
                    precision_blend_lambda=p["precision_blend_lambda"],
                    drift_penalty_weight=p["drift_penalty_weight"],
                    use_perturbation=use_perturbation,
                    perturbation_mode=perturbation_mode,
                    n_perturb=n_perturb,
                    X_min=X_min, X_range=X_range,
                    rng=np.random.default_rng(seed + instance_idx * 1000),
                    x_star_unit=None,
                    initial_window=0.1,
                    js_penalty_weight=p["js_penalty_weight"],
                )
            
            info_g, rule_g, lower_g, upper_g = greedy_rollout(env, policy, device, rl_steps_per_episode)
            
            env.lower[:] = lower_g
            env.upper[:] = upper_g
            prec_check, cov_check, det_check = env._current_metrics()
            
            all_anchor_results.append({
                "precision": float(prec_check),
                "hard_precision": float(det_check.get("hard_precision", 0.0)),
                "coverage": float(cov_check),
                "rule": rule_g,
                "lower": lower_g.tolist(),
                "upper": upper_g.tolist(),
                "instance_idx": int(instance_idx),
            })
        
        print()
        
        if len(all_anchor_results) > 0:
            avg_prec = float(np.mean([r["precision"] for r in all_anchor_results]))
            avg_hard_prec = float(np.mean([r["hard_precision"] for r in all_anchor_results]))
            avg_cov = float(np.mean([r["coverage"] for r in all_anchor_results]))
            best_anchor = max(all_anchor_results, key=lambda r: r["hard_precision"])
            final_greedy[int(cls)] = {
                "precision": avg_prec,
                "hard_precision": avg_hard_prec,
                "coverage": avg_cov,
                "rule": best_anchor["rule"],
                "lower": best_anchor["lower"],
                "upper": best_anchor["upper"],
                "num_instances": len(sel),
                "num_rollouts": len(all_anchor_results),
                "total_anchors": len(all_anchor_results),
            }
            final_greedy_all[int(cls)] = all_anchor_results
            print(f"[greedy cls={cls}] Completed: avg_precision={avg_hard_prec:.3f}, avg_coverage={avg_cov:.3f}, n={len(all_anchor_results)}")
    
    # Final confusion matrix
    classifier.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X_test).float().to(device)
        probs_final = classifier(inputs).cpu().numpy()
    final_preds = probs_final.argmax(axis=1)
    cm = confusion_matrix(y_test, final_preds, labels=list(range(num_classes)))
    
    if show_plots:
        fig = plt.figure(figsize=(6, 5))
        plt.imshow(cm, interpolation='nearest')
        plt.title('Confusion Matrix (test set)')
        plt.colorbar()
        plt.xlabel('Predicted label')
        plt.ylabel('True label')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, str(cm[i, j]), ha='center', va='center', fontsize=8)
        plt.tight_layout()
        plt.show()
    
    # Plot training histories
    if show_plots:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        axes[0].plot(clf_history['train_accs'], label='train')
        axes[0].plot(clf_history['test_accs'], label='test')
        axes[0].set_title('Classifier Accuracy')
        axes[0].set_xlabel('Epoch')
        axes[0].set_ylabel('Accuracy')
        axes[0].legend()
        
        axes[1].plot(rl_history['episode_rewards'])
        axes[1].set_title('RL Episode Returns')
        axes[1].set_xlabel('Episode')
        axes[1].set_ylabel('Return')
        plt.tight_layout()
        plt.show()
    
    return {
        "classifier": classifier,
        "policy": policy,
        "value_fn": value_fn,
        "classifier_history": clf_history,
        "rl_history": rl_history,
        "final_greedy": final_greedy,
        "final_greedy_all": final_greedy_all,
        "metadata": {
            "dataset": dataset,
            "num_classes": num_classes,
            "feature_names": feature_names,
            "class_names": class_names,
            "n_features": int(X_train.shape[1]),
            "seed": seed,
            "classifier_epochs": classifier_epochs,
            "rl_episodes": rl_episodes,
            "rl_steps_per_episode": rl_steps_per_episode,
            "use_discretization": use_discretization,
            "bin_edges": [e.tolist() if isinstance(e, np.ndarray) else e for e in edges] if (use_discretization and edges is not None) else None,
            "scaler_mean": scaler.mean_.tolist(),
            "scaler_scale": scaler.scale_.tolist(),
            "X_min": X_min.tolist(),
            "X_range": X_range.tolist(),
            "preset_params": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in p.items()},
        }
    }


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run post-hoc dynamic anchor training.")
    parser.add_argument("--dataset", type=str, default="covtype", choices=["breast_cancer", "synthetic", "covtype"], help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=["auto", "cuda", "mps", "cpu"], help="Device: auto|cuda|mps|cpu")
    parser.add_argument("--classifier_epochs", type=int, default=100, help="Classifier max epochs")
    parser.add_argument("--classifier_batch_size", type=int, default=256, help="Classifier batch size")
    parser.add_argument("--classifier_lr", type=float, default=1e-3, help="Classifier learning rate")
    parser.add_argument("--classifier_patience", type=int, default=10, help="Classifier early stopping patience")
    parser.add_argument("--rl_episodes", type=int, default=None, help="RL episodes (None=use dataset-specific defaults)")
    parser.add_argument("--rl_steps", type=int, default=None, help="RL steps per episode (None=use dataset-specific defaults)")
    parser.add_argument("--rl_ppo_epochs", type=int, default=4, help="PPO update epochs")
    parser.add_argument("--rl_clip_epsilon", type=float, default=0.2, help="PPO clipping parameter")
    parser.add_argument("--rl_entropy_coef", type=float, default=None, help="RL entropy coefficient (None=use dataset-specific defaults)")
    parser.add_argument("--rl_value_coef", type=float, default=0.5, help="RL value coefficient")
    parser.add_argument("--rl_batch_size", type=int, default=None, help="RL batch size (None = all experiences)")
    parser.add_argument("--rl_lr", type=float, default=3e-4, help="RL learning rate")
    parser.add_argument("--use_perturbation", action="store_true", help="Enable perturbation-based sampling inside boxes")
    parser.add_argument("--no-perturbation", dest="use_perturbation", action="store_false", help="Disable perturbation-based sampling inside boxes")
    parser.set_defaults(use_perturbation=None)  # None means use dataset-specific defaults
    parser.add_argument("--perturbation_mode", type=str, default=None, choices=["bootstrap", "uniform"], help="Perturbation mode (None=use dataset-specific defaults)")
    parser.add_argument("--n_perturb", type=int, default=None, help="Number of perturbations (None=use dataset-specific defaults)")
    parser.add_argument("--show_plots", action="store_true", default=False, help="Show plots (default: False)")
    parser.add_argument("--num_greedy_rollouts", type=int, default=20, help="Number of greedy rollouts per class")
    parser.add_argument("--max_features_in_rule", type=int, default=5, help="Max features in rule")
    
    args = parser.parse_args()
    
    results = train_posthoc_dynamic_anchors(
        dataset=args.dataset,
        seed=args.seed,
        device_preference=args.device,
        classifier_epochs=args.classifier_epochs,
        classifier_batch_size=args.classifier_batch_size,
        classifier_lr=args.classifier_lr,
        classifier_patience=args.classifier_patience,
        rl_episodes=args.rl_episodes,
        rl_steps_per_episode=args.rl_steps,
        rl_ppo_epochs=args.rl_ppo_epochs,
        rl_clip_epsilon=args.rl_clip_epsilon,
        rl_entropy_coef=args.rl_entropy_coef,
        rl_value_coef=args.rl_value_coef,
        rl_batch_size=args.rl_batch_size,
        rl_lr=args.rl_lr,
        use_perturbation=args.use_perturbation,
        perturbation_mode=args.perturbation_mode,
        n_perturb=args.n_perturb,
        show_plots=args.show_plots,
        num_greedy_rollouts=args.num_greedy_rollouts,
        max_features_in_rule=args.max_features_in_rule,
    )
    
    # Save models and results - use resolved values from metadata
    rl_episodes_resolved = results.get('metadata', {}).get('rl_episodes', args.rl_episodes)
    rl_steps_resolved = results.get('metadata', {}).get('rl_steps_per_episode', args.rl_steps)
    model_prefix = f'posthoc_{args.dataset}_{args.seed}_{results.get("metadata", {}).get("classifier_epochs", args.classifier_epochs)}_{rl_episodes_resolved}_{rl_steps_resolved}'
    
    if results.get('classifier') is not None:
        classifier_file = f'classifier_{model_prefix}.pth'
        torch.save(results['classifier'].state_dict(), classifier_file)
        print(f"Classifier saved to {classifier_file}")
    
    if results.get('policy') is not None:
        policy_file = f'policy_{model_prefix}.pth'
        torch.save(results['policy'].state_dict(), policy_file)
        print(f"Policy saved to {policy_file}")
    
    if results.get('value_fn') is not None:
        value_file = f'value_fn_{model_prefix}.pth'
        torch.save(results['value_fn'].state_dict(), value_file)
        print(f"Value network saved to {value_file}")
    
    # Save results as JSON (excluding PyTorch models)
    results_copy = results.copy()
    results_copy.pop('classifier', None)
    results_copy.pop('policy', None)
    results_copy.pop('value_fn', None)
    
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return float(obj)
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, dict):
            return {key: convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_to_serializable(item) for item in obj]
        else:
            return obj
    
    results_serializable = convert_to_serializable(results_copy)
    
    results_file = f'results_{model_prefix}.json'
    with open(results_file, 'w') as f:
        json.dump(results_serializable, f, indent=4)
    print(f"Results saved to {results_file}")

