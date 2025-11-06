import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer, fetch_covtype
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score


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


DEVICE_CHOICES = ("auto", "cuda", "mps", "cpu")

def select_device(device_preference: str = "auto") -> torch.device:
    device_preference = (device_preference or "auto").lower()
    if device_preference == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device_preference == "mps":
        return torch.device("mps" if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else "cpu")
    if device_preference == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_dataset(dataset: str, seed: int):
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
        raise ValueError(f"Unknown dataset '{dataset}'. Choose 'breast_cancer', 'synthetic', or 'covtype'.")

    # Normalize class labels to 0..C-1 and prepare class names aligned with indices
    unique_classes = np.unique(y)
    if dataset == "breast_cancer":
        class_names = [str(load_breast_cancer().target_names[c]) for c in unique_classes]
    else:
        class_names = [f"class_{int(c)}" for c in unique_classes]
    class_to_idx = {c: i for i, c in enumerate(unique_classes)}
    y = np.array([class_to_idx[c] for c in y], dtype=int)
    num_classes = int(len(unique_classes))

    return X, y, feature_names, class_names, num_classes


def train_classifier(X_train: np.ndarray, y_train: np.ndarray, input_dim: int, num_classes: int, device: torch.device,
                     epochs: int = 10, batch_size: int = 256):
    import math
    model = SimpleClassifier(input_dim, num_classes).to(device)
    opt = optim.Adam(model.parameters(), lr=1e-3)
    ce = nn.CrossEntropyLoss()

    X_tensor = torch.from_numpy(X_train).float()
    y_tensor = torch.from_numpy(y_train).long()

    last_loss = None
    last_train_acc = None
    model.train()
    n = X_train.shape[0]
    for _ in range(epochs):
        perm = torch.randperm(n)
        epoch_loss_sum = 0.0
        epoch_correct = 0
        epoch_count = 0
        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            xb = X_tensor[idx].to(device)
            yb = y_tensor[idx].to(device)
            opt.zero_grad()
            logits = model(xb)
            loss = ce(logits, yb)
            loss.backward()
            opt.step()
            epoch_loss_sum += float(loss.item()) * yb.size(0)
            epoch_count += int(yb.size(0))
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                epoch_correct += int((preds == yb).sum().item())
        last_loss = epoch_loss_sum / max(1, epoch_count)
        last_train_acc = epoch_correct / max(1, epoch_count)
        print(f"[clf-static] epoch {_+1}/{epochs} | loss={last_loss:.4f} | train_acc={last_train_acc:.3f} | samples={epoch_count}")
    return model, float(last_loss)


def evaluate(model: nn.Module, X: np.ndarray, y: np.ndarray, device: torch.device) -> float:
    model.eval()
    with torch.no_grad():
        inputs = torch.from_numpy(X).float().to(device)
        preds = model(inputs).argmax(dim=1).cpu().numpy()
    return float(accuracy_score(y, preds))


def run_static_anchors(
    dataset: str = "covtype",
    seed: int = 42,
    device_preference: str = "auto",
    anchor_threshold: float | None = None,
    num_instances_per_class: int | None = None,
    classifier_epochs: int | None = None,
    disc_perc: list[int] | None = None,
    coverage_target: float | None = None,
    max_anchor_size: int | None = None,
):
    rng = np.random.default_rng(seed)
    torch.manual_seed(seed)

    X, y, feature_names, class_names, num_classes = load_dataset(dataset, seed)
    print("*****************")
    print("")
    print("Run configuration")
    print("")
    print("*****************")
    print(f"[data] classes ({num_classes}): {class_names} | feature_names ({len(feature_names)}): {feature_names}")

    # Dataset-specific presets
    presets = {
        "breast_cancer": {
            "anchor_threshold": 0.95,
            "num_instances_per_class": 20,
            "classifier_epochs": 12,
            "disc_perc": [25, 50, 75],
            "coverage_target": 0.05,
        },
        "synthetic": {
            "anchor_threshold": 0.95,
            "num_instances_per_class": 30,
            "classifier_epochs": 10,
            "disc_perc": [20, 40, 60, 80],
            "coverage_target": 0.04,
        },
        "covtype": {
            "anchor_threshold": 0.95,
            "num_instances_per_class": 50,
            "classifier_epochs": 8,
            "disc_perc": [10, 25, 50, 75, 90],
            "coverage_target": 0.02,
        },
    }
    p = presets[dataset]

    # Resolve None to dataset defaults
    anchor_threshold = float(anchor_threshold if anchor_threshold is not None else p["anchor_threshold"]) 
    num_instances_per_class = int(num_instances_per_class if num_instances_per_class is not None else p["num_instances_per_class"]) 
    classifier_epochs = int(classifier_epochs if classifier_epochs is not None else p["classifier_epochs"]) 
    disc_perc = list(disc_perc if disc_perc is not None else p["disc_perc"]) 
    # Resolve coverage target
    coverage_target = float(coverage_target if coverage_target is not None else p["coverage_target"]) 
    print(f"[auto] using dataset-specific defaults: threshold={anchor_threshold}, per_class={num_instances_per_class}, clf_epochs={classifier_epochs}, disc_perc={disc_perc}, coverage_target={coverage_target}")

    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw).astype(np.float32)
    X_test = scaler.transform(X_test_raw).astype(np.float32)

    device = select_device(device_preference)
    print(f"[device] using {device}")

    model, _ = train_classifier(X_train, y_train, X_train.shape[1], num_classes, device, epochs=classifier_epochs)
    test_acc = evaluate(model, X_test, y_test, device)
    print(f"[clf] test_acc={test_acc:.3f}")

    # Anchor-Exp (marcotcr/anchor) setup
    try:
        from anchor import anchor_tabular
    except Exception as e:
        raise RuntimeError("anchor-exp is required for static anchors. Install with `pip install anchor-exp`.") from e

    def predict_labels(x: np.ndarray) -> np.ndarray:
        model.eval()
        with torch.no_grad():
            t = torch.from_numpy(x.astype(np.float32)).to(device)
            preds = model(t).argmax(dim=1).cpu().numpy()
        return preds

    categorical_names = {}
    # Note: AnchorTabularExplainer expects train_data as positional arg
    explainer = anchor_tabular.AnchorTabularExplainer(
        class_names,
        feature_names,
        X_train,
        categorical_names,
    )
    # Fit with train and validation sets as required by anchor-exp API
    explainer.fit(X_train, y_train, X_test, y_test)

    # Try to extract the exact bin thresholds used by the anchor discretizer
    bin_edges: list[np.ndarray] = []
    try:
        disc = getattr(explainer, 'discretizer', None)
        n_features = X_train.shape[1]
        for j in range(n_features):
            edges_j = None
            # Common attribute names across versions
            for attr in ['thresholds_', 'thresholds', 'percentiles_', 'percentiles', 'bins_', 'bins']:
                if disc is not None and hasattr(disc, attr):
                    obj = getattr(disc, attr)
                    try:
                        # obj could be list-like per feature
                        cand = obj[j]
                        edges_j = np.array(cand, dtype=np.float32).ravel()
                        break
                    except Exception:
                        pass
            if edges_j is None:
                # Fallback to computing percentiles consistent with defaults used earlier
                # This fallback keeps flows working even if internals change
                edges_j = np.unique(np.percentile(X_train[:, j], disc_perc).astype(np.float32))
            bin_edges.append(edges_j)
    except Exception:
        # On any unexpected structure, fallback to computed edges
        bin_edges = [np.unique(np.percentile(X_train[:, j], disc_perc).astype(np.float32)) for j in range(X_train.shape[1])]

    results = {c: [] for c in range(num_classes)}

    for cls in range(num_classes):
        idx_cls = np.where(y_test == cls)[0]
        if idx_cls.size == 0:
            continue
        # Sample up to K instances of this class
        sel = rng.choice(idx_cls, size=min(num_instances_per_class, idx_cls.size), replace=False)
        for i in sel:
            exp = explainer.explain_instance(
                X_test[i], 
                predict_labels, 
                threshold=anchor_threshold,
                max_anchor_size=max_anchor_size,
            )

            def _metric(val):
                try:
                    return float(val() if callable(val) else val)
                except Exception:
                    return 0.0

            # Extract precision/coverage accommodating callable API variants
            prec_val = _metric(getattr(exp, 'precision', 0.0))
            cov_val = _metric(getattr(exp, 'coverage', 0.0))

            # Extract human-readable anchor rule
            anchor_names = []
            if hasattr(exp, 'names'):
                names_attr = getattr(exp, 'names')
                try:
                    anchor_names = list(names_attr() if callable(names_attr) else names_attr)
                except Exception:
                    anchor_names = []
            elif hasattr(exp, 'as_list'):
                try:
                    anchor_names = list(exp.as_list())
                except Exception:
                    anchor_names = []

            results[cls].append({
                "precision": prec_val,
                "coverage": cov_val,
                "anchor": anchor_names,
            })
        if len(results[cls]) > 0:
            avg_prec = float(np.mean([r["precision"] for r in results[cls]]))
            avg_cov = float(np.mean([r["coverage"] for r in results[cls]]))
            print(f"[anchor cls={cls}] {class_names[cls] if cls < len(class_names) else cls} | avg_precision={avg_prec:.3f} | avg_coverage={avg_cov:.3f} | n={len(results[cls])}")

    return {
        "test_accuracy": test_acc,
        "per_class_results": results,
        "class_names": class_names,
        "feature_names": feature_names,
        "coverage_target": coverage_target,
        "bin_edges": bin_edges,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run static anchor baseline (alibi AnchorTabular)")
    parser.add_argument("--dataset", type=str, default="covtype", choices=["breast_cancer", "synthetic", "covtype"], help="Dataset to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="auto", choices=list(DEVICE_CHOICES), help="Device: auto|cuda|mps|cpu")
    parser.add_argument("--threshold", type=float, default=None, help="Anchor precision threshold (None=auto)")
    parser.add_argument("--per_class", type=int, default=None, help="Instances per class to explain (None=auto)")
    parser.add_argument("--classifier_epochs", type=int, default=None, help="Classifier training epochs (None=auto)")
    parser.add_argument("--disc_perc", type=int, nargs='*', default=None, help="Discretization percentiles (None=auto)")

    args = parser.parse_args()

    run_static_anchors(
        dataset=args.dataset,
        seed=args.seed,
        device_preference=args.device,
        anchor_threshold=args.threshold,
        num_instances_per_class=args.per_class,
        classifier_epochs=args.classifier_epochs,
        disc_perc=args.disc_perc,
    )
