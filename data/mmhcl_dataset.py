from __future__ import annotations
import hashlib
import json
from pathlib import Path
from typing import Optional
import numpy as np
import scipy.sparse as sp
import torch

def _describe_llm_artifact(path) -> str:
    try:
        from llm_augment.provenance import describe_artifact
        return describe_artifact(path)
    except Exception:
        return 'provenance unavailable (llm_augment not importable here)'
    except Exception:
        return 'provenance unavailable'
import numpy as np
import scipy.sparse as sp
import torch

def _sparse_cat_coo(H1: torch.Tensor, H2: torch.Tensor, dim: int=1) -> torch.Tensor:
    assert H1.is_sparse and H2.is_sparse, 'Both inputs must be sparse tensors'
    assert dim in (0, 1), 'Only dim=0 or dim=1 supported'
    H1 = H1.coalesce()
    H2 = H2.coalesce()
    idx1 = H1.indices()
    idx2 = H2.indices()
    val1 = H1.values()
    val2 = H2.values()
    if dim == 1:
        idx2_shifted = idx2.clone()
        idx2_shifted[1] += H1.shape[1]
        new_indices = torch.cat([idx1, idx2_shifted], dim=1)
        new_shape = (H1.shape[0], H1.shape[1] + H2.shape[1])
    else:
        idx2_shifted = idx2.clone()
        idx2_shifted[0] += H1.shape[0]
        new_indices = torch.cat([idx1, idx2_shifted], dim=1)
        new_shape = (H1.shape[0] + H2.shape[0], H1.shape[1])
    new_values = torch.cat([val1, val2])
    return torch.sparse_coo_tensor(new_indices, new_values, new_shape).coalesce()

def _norm_sparse(adj: torch.Tensor, normalization: str='sym') -> torch.Tensor:
    adj = adj.coalesce()
    idx = adj.indices()
    val = adj.values()
    r, c = (idx[0], idx[1])
    N0, N1 = (adj.shape[0], adj.shape[1])

    def _rowsum():
        rs = torch.zeros(N0, dtype=val.dtype, device=val.device)
        rs.scatter_add_(0, r, val)
        return rs
    if normalization == 'sym':
        d = torch.pow(_rowsum(), -0.5)
        d[torch.isinf(d)] = 0.0
        new_val = d[r] * val * d[c]
    elif normalization == 'rw':
        d = torch.pow(_rowsum(), -1)
        d[torch.isinf(d)] = 0.0
        new_val = d[r] * val
    elif normalization == 'origin':
        new_val = val
    else:
        raise ValueError(f'unknown normalization {normalization}')
    return torch.sparse_coo_tensor(idx, new_val, adj.shape).coalesce()

class MMHCLDataset:
    DATASETS = {'Clothing': {'modalities': ['image', 'text']}, 'Sports': {'modalities': ['image', 'text']}}

    def __init__(self, dataset: str='Clothing', mmhcl_dir: str='./data/MMHCL', core: str='5-core', cache_hypergraph: bool=True, knn_topk: int=10, knn_min_sim: float=0.0, modality_feature_files: Optional[dict[str, str]]=None, use_user_profile: bool=True, user_profile_file: str='user_profile_feat.npy', use_item_llm_text: bool=False, item_llm_text_file: str='text_llm_feat.npy'):
        assert dataset in self.DATASETS, f"Unknown dataset '{dataset}'. Choose from {list(self.DATASETS)}"
        self.dataset = dataset
        self.root = Path(mmhcl_dir) / dataset
        self.core_dir = self.root / core
        self.cache_hypergraph = cache_hypergraph
        self.knn_topk = knn_topk
        self.knn_min_sim = knn_min_sim
        self.modality_feature_files = dict(modality_feature_files or {})
        self.use_user_profile = bool(use_user_profile)
        self.user_profile_file = user_profile_file
        self.user_profile_feat: Optional[np.ndarray] = None
        self.use_item_llm_text = bool(use_item_llm_text)
        self.item_llm_text_file = item_llm_text_file
        self.item_llm_text_feat: Optional[np.ndarray] = None
        self.H_I_text: Optional[torch.Tensor] = None
        self.H_I_image: Optional[torch.Tensor] = None
        self.H_I_text_incidence: Optional[torch.Tensor] = None
        self.H_I_image_incidence: Optional[torch.Tensor] = None
        self._load_interactions()
        self._load_features()
        self._build_interaction_matrix()
        self._build_hypergraphs()

    def _load_interactions(self):

        def _read(name):
            p = self.core_dir / f'{name}.json'
            with open(p) as f:
                return json.load(f)
        train_data = _read('train')
        val_data = _read('val')
        test_data = _read('test')
        train_users = sorted((u for u, items in train_data.items() if items), key=lambda x: int(x))
        self.user2id = {u: i for i, u in enumerate(train_users)}
        self.num_users = len(self.user2id)
        self.raw_user_ids = [int(u) for u in train_users]
        val_data = {u: items for u, items in val_data.items() if u in self.user2id}
        test_data = {u: items for u, items in test_data.items() if u in self.user2id}
        max_item = 0
        for d in (train_data, val_data, test_data):
            for items in d.values():
                if items:
                    max_item = max(max_item, max(items))
        self.num_items = max_item + 1
        self.train_pairs: list[tuple[int, int]] = []
        for u_str, items in train_data.items():
            if u_str not in self.user2id:
                continue
            uid = self.user2id[u_str]
            for iid in items:
                self.train_pairs.append((uid, iid))
        self.train_data = train_data
        self.val_data = val_data
        self.test_data = test_data
        print(f'[MMHCLDataset] {self.dataset}: {self.num_users} users, {self.num_items} items, {len(self.train_pairs)} train pairs')

    def _load_features(self):
        modalities = self.DATASETS[self.dataset]['modalities']
        parts = []
        for mod in modalities:
            feat_name = self.modality_feature_files.get(mod, f'{mod}_feat.npy')
            path = self.root / feat_name
            feat = np.load(str(path)).astype(np.float32)
            norms = np.linalg.norm(feat, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            parts.append(feat / norms)
            setattr(self, f'item_{mod}_feat', feat / norms)
        for mod in modalities:
            feat = getattr(self, f'item_{mod}_feat', None)
            if feat is not None:
                if feat.shape[0] < self.num_items:
                    pad = np.zeros((self.num_items - feat.shape[0], feat.shape[1]), dtype=np.float32)
                    feat = np.concatenate([feat, pad], axis=0)
                elif feat.shape[0] > self.num_items:
                    feat = feat[:self.num_items]
                setattr(self, f'item_{mod}_feat', feat)
        for mod in modalities:
            feat = getattr(self, f'item_{mod}_feat', None)
            if feat is not None:
                zero_mask = np.linalg.norm(feat, axis=1) < 1e-08
                if zero_mask.any() and (~zero_mask).any():
                    feat[zero_mask] = feat[~zero_mask].mean(axis=0)
                    setattr(self, f'item_{mod}_feat', feat)
        self._load_user_profile_features()
        self._load_item_llm_text_features()

    def _load_item_llm_text_features(self):
        self.item_llm_text_feat = None
        if not self.use_item_llm_text:
            return
        path = self.root / self.item_llm_text_file
        if not path.exists():
            return
        feat = np.load(str(path)).astype(np.float32)
        if feat.shape[0] < self.num_items:
            pad = np.zeros((self.num_items - feat.shape[0], feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > self.num_items:
            feat = feat[:self.num_items]
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.item_llm_text_feat = feat / norms

    def _load_user_profile_features(self):
        self.user_profile_feat = None
        if not self.use_user_profile:
            return
        path = self.root / self.user_profile_file
        if not path.exists():
            return
        feat = np.load(str(path)).astype(np.float32)
        raw_ids = getattr(self, 'raw_user_ids', None)
        if raw_ids is not None:
            d = feat.shape[1]
            aligned = np.zeros((self.num_users, d), dtype=np.float32)
            for i, raw in enumerate(raw_ids):
                if 0 <= raw < feat.shape[0]:
                    aligned[i] = feat[raw]
            feat = aligned
        elif feat.shape[0] < self.num_users:
            pad = np.zeros((self.num_users - feat.shape[0], feat.shape[1]), dtype=np.float32)
            feat = np.concatenate([feat, pad], axis=0)
        elif feat.shape[0] > self.num_users:
            feat = feat[:self.num_users]
        norms = np.linalg.norm(feat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.user_profile_feat = feat / norms

    def _build_interaction_matrix(self):
        rows, cols = zip(*self.train_pairs)
        data = np.ones(len(rows), dtype=np.float32)
        self.R = sp.csr_matrix((data, (rows, cols)), shape=(self.num_users, self.num_items), dtype=np.float32)
        self._R_dense_cache = None

    def _get_dense_interaction_matrix(self) -> np.ndarray:
        if self._R_dense_cache is None:
            self._R_dense_cache = self.R.toarray()
        return self._R_dense_cache

    def build_R_normalized(self) -> torch.Tensor:
        if hasattr(self, '_R_normalized_cache') and self._R_normalized_cache is not None:
            return self._R_normalized_cache
        if not self.train_pairs:
            raise RuntimeError('train_pairs is empty — cannot build R_normalized.')
        pairs = torch.tensor(self.train_pairs, dtype=torch.long)
        u_idx, i_idx = (pairs[:, 0], pairs[:, 1])
        deg_u = torch.zeros(self.num_users, dtype=torch.float)
        deg_i = torch.zeros(self.num_items, dtype=torch.float)
        deg_u.scatter_add_(0, u_idx, torch.ones_like(u_idx, dtype=torch.float))
        deg_i.scatter_add_(0, i_idx, torch.ones_like(i_idx, dtype=torch.float))
        inv_sqrt_u = deg_u.clamp(min=1.0).rsqrt()
        inv_sqrt_i = deg_i.clamp(min=1.0).rsqrt()
        weights = inv_sqrt_u[u_idx] * inv_sqrt_i[i_idx]
        indices = torch.stack([u_idx, i_idx], dim=0)
        with torch.sparse.check_sparse_tensor_invariants(False):
            R = torch.sparse_coo_tensor(indices, weights, size=(self.num_users, self.num_items)).coalesce()
        self._R_normalized_cache = R
        return R

    def build_UI_mat(self, norm_type: str='sym') -> torch.Tensor:
        if getattr(self, '_UI_mat_cache', None) is not None:
            return self._UI_mat_cache
        cache_dir = self.root / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f'UI_mat_{norm_type}.pt'
        n = self.num_users + self.num_items
        if self.cache_hypergraph and cache_file.exists():
            cached = torch.load(str(cache_file), map_location='cpu')
            if tuple(cached.shape) == (n, n):
                self._UI_mat_cache = cached.coalesce()
                return self._UI_mat_cache
        Rcoo = self.R.tocoo()
        rows = np.concatenate([Rcoo.row, Rcoo.col + self.num_users])
        cols = np.concatenate([Rcoo.col + self.num_users, Rcoo.row])
        data = np.concatenate([Rcoo.data, Rcoo.data])
        with torch.sparse.check_sparse_tensor_invariants(False):
            A = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([rows, cols]).astype(np.int64)), torch.from_numpy(data.astype(np.float32)), (n, n)).coalesce()
        UI_mat = _norm_sparse(A, norm_type)
        if self.cache_hypergraph:
            torch.save(UI_mat.cpu(), str(cache_file))
        self._UI_mat_cache = UI_mat
        return UI_mat

    def build_U2U_mat(self, norm_type: str='rw') -> torch.Tensor:
        if getattr(self, '_U2U_mat_cache', None) is not None:
            return self._U2U_mat_cache
        cache_dir = self.root / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f'U2U_mat_{norm_type}.pt'
        if self.cache_hypergraph and cache_file.exists():
            cached = torch.load(str(cache_file), map_location='cpu')
            if tuple(cached.shape) == (self.num_users, self.num_users):
                self._U2U_mat_cache = cached.coalesce()
                return self._U2U_mat_cache
        Rs = self.R.tocsr().astype(np.float32)
        UU = (Rs @ Rs.T).tocoo()
        mask = UU.row != UU.col
        with torch.sparse.check_sparse_tensor_invariants(False):
            A = torch.sparse_coo_tensor(torch.from_numpy(np.vstack([UU.row[mask], UU.col[mask]]).astype(np.int64)), torch.from_numpy(UU.data[mask].astype(np.float32)), (self.num_users, self.num_users)).coalesce()
        U2U_mat = _norm_sparse(A, norm_type)
        if self.cache_hypergraph:
            torch.save(U2U_mat.cpu(), str(cache_file))
        self._U2U_mat_cache = U2U_mat
        return U2U_mat

    def build_I2I_mat(self, norm_type: str='sym') -> torch.Tensor:
        if getattr(self, '_I2I_mat_cache', None) is not None:
            return self._I2I_mat_cache
        cache_dir = self.root / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f'I2I_mat_mul_{norm_type}.pt'
        if self.cache_hypergraph and cache_file.exists():
            cached = torch.load(str(cache_file), map_location='cpu')
            if tuple(cached.shape) == (self.num_items, self.num_items):
                self._I2I_mat_cache = cached.coalesce()
                return self._I2I_mat_cache
        graphs = [g for g in (self.H_I_image, self.H_I_text) if g is not None and g.shape[0] == g.shape[1] == self.num_items]
        if not graphs:
            raise RuntimeError('build_I2I_mat needs H_I_text/H_I_image (modal kNN graphs) — none available.')
        H = graphs[0]
        for g in graphs[1:]:
            H = _sparse_cat_coo(H, g.coalesce(), dim=1)
        M = torch.sparse.mm(H, H.transpose(0, 1)).coalesce()
        I2I_mat = _norm_sparse(M, norm_type)
        if self.cache_hypergraph:
            torch.save(I2I_mat.cpu(), str(cache_file))
        self._I2I_mat_cache = I2I_mat
        return I2I_mat

    def _modal_adj_norm(self, H: torch.Tensor, tag: str) -> torch.Tensor:
        cache_dir = self.root / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        cache_file = cache_dir / f'modal_adj_{tag}_sym.pt'
        if self.cache_hypergraph and cache_file.exists():
            cached = torch.load(str(cache_file), map_location='cpu')
            if tuple(cached.shape) == (self.num_items, self.num_items):
                return cached.coalesce()
        Hc = H.coalesce()
        sym = torch.sparse_coo_tensor(torch.cat([Hc.indices(), Hc.indices().flip(0)], dim=1), torch.cat([Hc.values(), Hc.values()]), (self.num_items, self.num_items)).coalesce()
        adj = _norm_sparse(sym, 'sym')
        if self.cache_hypergraph:
            torch.save(adj.cpu(), str(cache_file))
        return adj

    def build_image_adj(self) -> torch.Tensor:
        if getattr(self, '_image_adj_cache', None) is not None:
            return self._image_adj_cache
        if self.H_I_image is None:
            raise RuntimeError('build_image_adj needs H_I_image (visual kNN).')
        self._image_adj_cache = self._modal_adj_norm(self.H_I_image, 'image')
        return self._image_adj_cache

    def build_text_adj(self) -> torch.Tensor:
        if getattr(self, '_text_adj_cache', None) is not None:
            return self._text_adj_cache
        if self.H_I_text is None:
            raise RuntimeError('build_text_adj needs H_I_text (textual kNN).')
        self._text_adj_cache = self._modal_adj_norm(self.H_I_text, 'text')
        return self._text_adj_cache

    def build_R_norm(self) -> torch.Tensor:
        return self.build_R_normalized()

    def build_R_row_norm(self) -> torch.Tensor:
        if not hasattr(self, '_R_row_norm_cache') or self._R_row_norm_cache is None:
            rows, cols, vals = ([], [], [])
            deg_u = {}
            for u, i in self.train_pairs:
                deg_u[u] = deg_u.get(u, 0) + 1
            for u, i in self.train_pairs:
                rows.append(u)
                cols.append(i)
                vals.append(1.0 / deg_u[u])
            idx = torch.tensor([rows, cols], dtype=torch.long)
            v = torch.tensor(vals, dtype=torch.float32)
            R = torch.sparse_coo_tensor(idx, v, (self.num_users, self.num_items)).coalesce()
            self._R_row_norm_cache = R
        return self._R_row_norm_cache

    def _build_hypergraphs(self):
        cache_dir = self.root / 'cache'
        cache_dir.mkdir(exist_ok=True, parents=True)
        H_U_file = cache_dir / 'H_U_binary.pt'
        H_I_file = cache_dir / 'H_I_binary.pt'
        from data.hypergraph_builder import U2UHypergraphBuilder, I2IHypergraphBuilder
        expected_HU = (self.num_users, self.num_items)
        expected_HI = (self.num_items, self.num_users)
        cache_valid = False
        if self.cache_hypergraph and H_U_file.exists() and H_I_file.exists():
            self.H_U = torch.load(str(H_U_file), map_location='cpu')
            self.H_I = torch.load(str(H_I_file), map_location='cpu')
            if tuple(self.H_U.shape) == expected_HU and tuple(self.H_I.shape) == expected_HI:
                cache_valid = True
        if not cache_valid:
            u2u = U2UHypergraphBuilder(n_users=self.num_users, n_items=self.num_items, interaction_matrix=self._get_dense_interaction_matrix())
            self.H_U = u2u.build_incidence_matrix()
            i2i = I2IHypergraphBuilder(n_users=self.num_users, n_items=self.num_items, interaction_matrix=self._get_dense_interaction_matrix())
            self.H_I = i2i.build_incidence_matrix()
            if self.cache_hypergraph:
                torch.save(self.H_U.cpu(), str(H_U_file))
                torch.save(self.H_I.cpu(), str(H_I_file))
        self._build_modality_knn_hypergraphs(cache_dir)

    @staticmethod
    def _feature_fingerprint(feat: np.ndarray) -> str:
        digest = hashlib.blake2b(digest_size=8)
        digest.update(str(feat.shape).encode())
        digest.update(str(feat.dtype).encode())
        flat = np.ascontiguousarray(feat).reshape(-1)
        stride = max(1, flat.size // 65536)
        digest.update(flat[::stride].tobytes())
        return digest.hexdigest()

    def _build_modality_knn_hypergraphs(self, cache_dir: Path):
        from data.hypergraph_builder import build_modality_knn_hypergraph
        image_feat = getattr(self, 'item_image_feat', None)
        text_feat = getattr(self, 'item_text_feat', None)
        from pathlib import Path as _P
        _text_stem = _P(self.modality_feature_files.get('text', 'text_feat.npy')).stem
        _image_stem = _P(self.modality_feature_files.get('image', 'image_feat.npy')).stem
        if text_feat is not None:
            _text_stem = f'{_text_stem}_{self._feature_fingerprint(text_feat)}'
        if image_feat is not None:
            _image_stem = f'{_image_stem}_{self._feature_fingerprint(image_feat)}'
        if text_feat is not None:
            tag_text = f'{_text_stem}_topk{self.knn_topk}_minsim{self.knn_min_sim}'
            H_I_text_file = cache_dir / f'H_I_text_{tag_text}.pt'
            expected_text = (self.num_items, self.num_items)
            H_I_text = None
            if self.cache_hypergraph and H_I_text_file.exists():
                cached = torch.load(str(H_I_text_file), map_location='cpu')
                if tuple(cached.shape) == expected_text:
                    H_I_text = cached
            if H_I_text is None:
                H_I_text = build_modality_knn_hypergraph(text_feat, top_k=self.knn_topk, min_sim_threshold=self.knn_min_sim)
                if self.cache_hypergraph:
                    torch.save(H_I_text.cpu(), str(H_I_text_file))
            self.H_I_text = H_I_text
        if image_feat is not None:
            tag_image = f'{_image_stem}_topk{self.knn_topk}_minsim{self.knn_min_sim}'
            H_I_image_file = cache_dir / f'H_I_image_{tag_image}.pt'
            expected_image = (self.num_items, self.num_items)
            H_I_image = None
            if self.cache_hypergraph and H_I_image_file.exists():
                cached = torch.load(str(H_I_image_file), map_location='cpu')
                if tuple(cached.shape) == expected_image:
                    H_I_image = cached
            if H_I_image is None:
                H_I_image = build_modality_knn_hypergraph(image_feat, top_k=self.knn_topk, min_sim_threshold=self.knn_min_sim)
                if self.cache_hypergraph:
                    torch.save(H_I_image.cpu(), str(H_I_image_file))
            self.H_I_image = H_I_image
        self._build_modal_incidence(cache_dir)

    def _build_modal_incidence(self, cache_dir):
        from data.hypergraph_builder import build_knn_cluster_incidence
        N = self.num_items
        n_clusters = max(int(N ** 0.5), 64)
        image_feat = getattr(self, 'item_image_feat', None)
        text_feat = getattr(self, 'item_text_feat', None)
        from pathlib import Path as _P
        _stem = {'image': _P(self.modality_feature_files.get('image', 'image_feat.npy')).stem, 'text': _P(self.modality_feature_files.get('text', 'text_feat.npy')).stem}
        for modality, feat, attr_name in [('image', image_feat, 'H_I_image_incidence'), ('text', text_feat, 'H_I_text_incidence')]:
            if feat is None:
                continue
            tag = f'{_stem[modality]}_{self._feature_fingerprint(feat)}'
            cache_file = cache_dir / f'H_I_{modality}_{tag}_incidence_k{n_clusters}.pt'
            if self.cache_hypergraph and cache_file.exists():
                H = torch.load(str(cache_file), map_location='cpu')
                if H.shape == (N, n_clusters):
                    setattr(self, attr_name, H)
                    continue
            H = build_knn_cluster_incidence(features=feat, top_k=10, n_clusters=n_clusters, min_sim=0.0, weighted=True)
            setattr(self, attr_name, H)
            if self.cache_hypergraph:
                torch.save(H.cpu(), str(cache_file))

    def get_val_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        for u_str, items in self.val_data.items():
            uid = self.user2id.get(u_str)
            if uid is None:
                continue
            for iid in items:
                pairs.append((uid, iid))
        return pairs

    def get_test_pairs(self) -> list[tuple[int, int]]:
        pairs = []
        for u_str, items in self.test_data.items():
            uid = self.user2id.get(u_str)
            if uid is None:
                continue
            for iid in items:
                pairs.append((uid, iid))
        return pairs
