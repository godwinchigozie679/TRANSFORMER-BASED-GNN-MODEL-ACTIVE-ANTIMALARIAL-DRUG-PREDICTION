"""Training one model (GNN or baseline), prediction, metrics and the validation-based cutoff."""
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, confusion_matrix, f1_score, roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from torch_geometric.loader import DataLoader

from config import CFG, DEVICE
from models import Net
from splits import undersample


def predict(model, loader):
    model.eval()
    out = []
    with torch.no_grad():
        for b in loader:
            out.append(torch.sigmoid(model(b.to(DEVICE)).float()).cpu().numpy())
    return np.concatenate(out)


def fit_gnn(ctx, fd, kind, params, epochs, patience, seed):
    """Trains one GNN with early stopping on validation AUROC; returns the best model and its predictions."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    model = Net(kind, fd, heads=CFG["heads"], **params).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=params["lr"], weight_decay=params["weight_decay"])
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, mode="max", factor=0.5, patience=5)
    use_amp = CFG["amp"] and DEVICE.type == "cuda"
    scaler = (torch.amp.GradScaler("cuda", enabled=use_amp) if hasattr(torch.amp, "GradScaler")
              else torch.cuda.amp.GradScaler(enabled=use_amp))
    rng = np.random.default_rng(seed)

    def epoch_loader():
        """A fresh 1:1 undersample of the training pool every epoch (so all inactives are used over time),
        or the single fixed undersample when resample_each_epoch is off."""
        idx = undersample(fd.pool, ctx.y, rng) if CFG["resample_each_epoch"] else fd.train
        return DataLoader([ctx.graphs[i] for i in idx], batch_size=CFG["batch_size"], shuffle=True,
                          num_workers=CFG["workers"], pin_memory=DEVICE.type == "cuda")

    va = [b.to(DEVICE) for b in DataLoader([ctx.graphs[i] for i in fd.val], batch_size=CFG["eval_batch_size"],
                                           shuffle=False)]      # collated once, reused every epoch
    yv = ctx.y[fd.val]
    best_auc, best_state, best_p, bad, ep = -1.0, None, None, 0, 0
    hist = {k: [] for k in ("train_loss", "val_loss", "train_auc", "val_auc", "train_acc", "val_acc", "lr")}
    for ep in range(1, epochs + 1):
        model.train()
        tl, tp, ty = [], [], []
        for b in epoch_loader():
            b = b.to(DEVICE)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type=DEVICE.type, dtype=torch.float16, enabled=use_amp):
                logit = model(b).float()
                loss = F.binary_cross_entropy_with_logits(logit, b.y.float())
            scaler.scale(loss).backward()
            if CFG["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), CFG["grad_clip"])
            scaler.step(opt)
            scaler.update()
            tl.append(loss.item() * len(b.y))
            tp.append(torch.sigmoid(logit.detach()).cpu().numpy())
            ty.append(b.y.cpu().numpy())
        p = predict(model, va)
        auc = roc_auc_score(yv, p)
        tp, ty = np.concatenate(tp), np.concatenate(ty)           # training metrics: running values during the epoch
        pc = np.clip(p, 1e-7, 1 - 1e-7)
        for k, v in (("train_loss", sum(tl) / len(ty)), ("val_loss", float(-np.mean(yv * np.log(pc) + (1 - yv) * np.log(1 - pc)))),
                     ("train_auc", roc_auc_score(ty, tp)), ("val_auc", auc),
                     ("train_acc", float(np.mean((tp >= 0.5) == ty))), ("val_acc", float(np.mean((p >= 0.5) == yv))),
                     ("lr", opt.param_groups[0]["lr"])):
            hist[k].append(float(v))
        sched.step(auc)
        if auc > best_auc + 1e-5:
            best_auc, best_p, bad = auc, p, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state, strict=False)
    model.history = hist                                       # per-epoch curves, saved by the final stage
    return model, best_auc, ep, best_p


def make_baseline(key, seed):
    return {
        "baseline_gradient_boosting": lambda: HistGradientBoostingClassifier(random_state=seed),
        "baseline_random_forest": lambda: RandomForestClassifier(n_jobs=-1, random_state=seed),
        "baseline_logistic_regression": lambda: LogisticRegression(max_iter=2000),
        "baseline_linear_svm": lambda: CalibratedClassifierCV(LinearSVC(max_iter=5000), method="sigmoid", cv=3),
        "baseline_knn": lambda: KNeighborsClassifier(n_neighbors=CFG["knn_k"], n_jobs=-1),
    }[key]()


def fit_baseline(ctx, fd, key, seed):
    model = make_baseline(key, seed).fit(fd.table[fd.train], ctx.y[fd.train])
    return model, model.predict_proba(fd.table[fd.val])[:, 1]


def metrics(y, p, thr=0.5):
    pred = (p >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    rec = tp / max(tp + fn, 1)
    spec = tn / max(tn + fp, 1)
    prec = tp / max(tp + fp, 1)
    denom = np.sqrt(float(tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(rec + spec) / 2, precision=prec, recall=rec,
                f1=2 * prec * rec / max(prec + rec, 1e-12), mcc=(tp * tn - fp * fn) / denom if denom else 0.0,
                auc=roc_auc_score(y, p), pr_auc=average_precision_score(y, p), TN=int(tn), FP=int(fp),
                FN=int(fn), TP=int(tp))


def best_cutoff(y, p):
    """Cutoff that maximises F1 on the validation split (never on the test set)."""
    grid = np.arange(0.05, 0.96, 0.01)
    scores = [f1_score(y, (p >= t).astype(int), zero_division=0) for t in grid]
    return float(round(grid[int(np.argmax(scores))], 2))
