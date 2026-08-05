"""
GDN (category ③, Sec 8.2) wrapped in this benchmark's fit/score
interface, reusing `test_gdn/gdn_model.GDN` directly rather than
reimplementing it -- this is the mandatory "structure signal alone, no
memory, no federation" ablation baseline every result in
`test_gdn_physi/` should be compared against (Sec 8.4, ablation #1).

Kept deliberately lightweight (short history, few epochs) for benchmark-
harness speed; not tuned for peak AUROC the way `test_gdn/train_gdn.py`
is -- use that script directly for the best-effort GDN number, this
wrapper is for apples-to-apples cross-baseline comparison runs.
"""
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[2]  # FLMemGraph itself, test_gdn/ now lives here
sys.path.insert(0, str(REPO_ROOT / "test_gdn"))
from gdn_model import GDN  # noqa: E402

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GDNBaseline:
    def __init__(self, history_len: int = 10, epochs: int = 15, embed_dim: int = 64, top_k: int = 15):
        self.history_len = history_len
        self.epochs = epochs
        self.embed_dim = embed_dim
        self.top_k = top_k
        self.model = None
        self.median = None
        self.iqr = None

    def _dense_pairs(self, windows):
        n, t, f = windows.shape
        h = self.history_len
        num_targets = t - h
        history = np.stack([windows[:, i : i + h] for i in range(num_targets)], axis=1)
        target = np.stack([windows[:, i + h] for i in range(num_targets)], axis=1)
        window_id = np.repeat(np.arange(n), num_targets)
        return (history.reshape(-1, h, f).astype(np.float32),
                target.reshape(-1, f).astype(np.float32), window_id)

    def _per_window_error(self, windows, batch_size=256):
        history, target, window_id = self._dense_pairs(windows)
        self.model.eval()
        errs = []
        with torch.no_grad():
            for i in range(0, len(history), batch_size):
                h = torch.from_numpy(history[i : i + batch_size]).to(DEVICE)
                t = torch.from_numpy(target[i : i + batch_size]).to(DEVICE)
                pred = self.model(h)
                errs.append(((pred - t) ** 2).cpu().numpy())
        err = np.concatenate(errs, axis=0)
        out = np.zeros((len(windows), err.shape[1]))
        counts = np.zeros(len(windows))
        np.add.at(out, window_id, err)
        np.add.at(counts, window_id, 1)
        return (out / counts[:, None]).astype(np.float32)

    def fit(self, fit_windows, calib_windows):
        torch.manual_seed(42)
        num_nodes = fit_windows.shape[2]
        self.model = GDN(num_nodes=num_nodes, window_size=self.history_len,
                          embed_dim=self.embed_dim, top_k=self.top_k).to(DEVICE)
        history, target, _ = self._dense_pairs(fit_windows)
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(torch.from_numpy(history), torch.from_numpy(target)),
            batch_size=256, shuffle=True,
        )
        optimizer = torch.optim.Adam(self.model.parameters(), lr=1e-3)
        criterion = torch.nn.MSELoss()
        for _ in range(self.epochs):
            self.model.train()
            for h, t in loader:
                h, t = h.to(DEVICE), t.to(DEVICE)
                optimizer.zero_grad()
                loss = criterion(self.model(h), t)
                loss.backward()
                optimizer.step()

        calib_err = self._per_window_error(calib_windows)
        self.median = np.median(calib_err, axis=0)
        q75, q25 = np.percentile(calib_err, [75, 25], axis=0)
        self.iqr = np.maximum(q75 - q25, 1e-8)
        return self

    def score(self, windows):
        err = self._per_window_error(windows)
        return ((err - self.median) / self.iqr).max(axis=1)
