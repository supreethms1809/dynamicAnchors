import torch
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainers.multiagent_networks import MultiAgentEnvironment, CentralizedCritic, \
                                    MultiAgentPolicyNet, DecentralizedExecutor, \
                                    CentralizedTrainer
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def normalize_obs(obs):
    return (obs - obs.mean(dim=0)) / obs.std(dim=0)

def normalize_action(action):
    return (action - action.mean(dim=0)) / action.std(dim=0)

def normalize_dataset(X, y):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    y = scaler.transform(y)
    return X, y

def find_centroids(X, y, class_name):
    X_class = X[y == class_name]
    kmeans = KMeans(n_clusters=10, random_state=42)
    kmeans.fit(X_class)
    return kmeans.cluster_centers_

def find_centroids_per_class(X, y):
    centroids_per_class = {}
    for class_name in np.unique(y):
        centroids_per_class[class_name] = find_centroids(X, y, class_name)
    return centroids_per_class

def find_window_for_instance(X, y, instance_index):
    X_instance = X[instance_index]
    y_instance = y[instance_index]
    centroids_per_class = find_centroids_per_class(X, y)
    for class_name, centroids in centroids_per_class.items():
        if y_instance == class_name:
            return {
                "lower_bound": centroids.min(axis=0),
                "upper_bound": centroids.max(axis=0)
            }
    return None

def initialize_initial_window(interpretation_type: str, X, y, instance_index=None):
    if interpretation_type == "class":
        cluster_centroids_per_class = find_centroids_per_class(X, y)
        initial_window = {}
        for class_name, centroids in cluster_centroids_per_class.items():
            initial_window[class_name] = {
                "lower_bound": centroids.min(axis=0),
                "upper_bound": centroids.max(axis=0)
            }
        return initial_window
    elif interpretation_type == "instance":
        initial_window = find_window_for_instance(X, y, instance_index)
        if initial_window is None:
            raise ValueError(f"No window found for instance {instance_index}")
        return initial_window
    else:
        raise ValueError(f"Invalid interpretation type: {interpretation_type}")