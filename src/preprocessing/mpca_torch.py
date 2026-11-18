from collections.abc import Callable, Iterable, Iterator

import torch


class TorchMPCA:
    def __init__(
        self,
        ranks: tuple[int, int, int],
        projection_mode: str = 'axial',
        max_iter: int = 15,
        tol: float = 1e-6,
        dtype: str | torch.dtype = 'float64',
        device: str | torch.device = 'cpu',
    ):
        if projection_mode not in {'axial', 'frontal', 'sagital', 'full'}: raise ValueError(f'Unsupported projection mode: {projection_mode}')

        self.dtype = self._resolve_dtype(dtype)
        self.device = torch.device(device)

        if self.device.type == 'cuda' and not torch.cuda.is_available(): raise RuntimeError('CUDA was requested but is unavailable')

        self.ranks = tuple(int(rank) for rank in ranks)
        
        if len(self.ranks) != 3 or any(rank < 1 for rank in self.ranks): raise ValueError('Expected three positive MPCA ranks')

        self.projection_mode = projection_mode
        self.max_iter = int(max_iter)
        self.tol = float(tol)
        self.bases = None
        self.eigenvalues = None
        self.orig_shape = None
        self.initialized = False
        self.iterations = 0

    @staticmethod
    def _resolve_dtype(dtype: str | torch.dtype) -> torch.dtype:
        if dtype in {'fp32', 'float32', torch.float32}:
            return torch.float32
        if dtype in {'fp64', 'float64', torch.float64}:
            return torch.float64
        raise ValueError('dtype must be fp32/float32 or fp64/float64')

    def _tensor(self, value) -> torch.Tensor:
        return torch.as_tensor(value, dtype=self.dtype, device=self.device)

    @staticmethod
    def _unfold(volume: torch.Tensor, mode: int) -> torch.Tensor:
        return volume.movedim(mode, 0).reshape(volume.shape[mode], -1)

    @staticmethod
    def _fold(matrix: torch.Tensor, mode: int, shape: tuple[int, ...]) -> torch.Tensor:
        moved_shape = list(shape)
        moved_shape.insert(0, moved_shape.pop(mode))
        return matrix.reshape(moved_shape).movedim(0, mode)

    @classmethod
    def _mode_product(
        cls,
        volume: torch.Tensor,
        matrix: torch.Tensor,
        mode: int,
    ) -> torch.Tensor:
        
        result = matrix @ cls._unfold(volume, mode)
        shape = list(volume.shape)
        shape[mode] = matrix.shape[0]

        return cls._fold(result, mode, tuple(shape))

    def _validate_batches(self, batches: Callable[[], Iterable[torch.Tensor]], n_samples: int, volume_shape: tuple[int, int, int]) -> None:
        if n_samples < 1 or len(volume_shape) != 3: raise ValueError('Expected a non-empty dataset of 3D volumes')

        seen = 0
        for batch in batches():
            tensor = self._tensor(batch)

            if tensor.ndim != 4 or tuple(tensor.shape[1:]) != tuple(volume_shape): raise ValueError('Batches must have shape (batch, X, Y, Z)')
            
            if not torch.isfinite(tensor).all(): raise ValueError('MPCA input contains non-finite values')
            seen += tensor.shape[0]

        if seen != n_samples:
            raise ValueError(f'Expected {n_samples} samples, received {seen}')

    def fit_batches(self, batches: Callable[[], Iterable[torch.Tensor]], n_samples: int, volume_shape: tuple[int, int, int]):
        self._validate_batches(batches, n_samples, volume_shape)
        shape = tuple(volume_shape)
        if any(rank > dim for rank, dim in zip(self.ranks, shape, strict=True)):
            raise ValueError('MPCA ranks cannot exceed input dimensions')

        bases = [torch.eye(dim, dtype=self.dtype, device=self.device)[:, :rank] for dim, rank in zip(shape, self.ranks, strict=True)]
        eigenvalue_sets = [None, None, None]

        original_variance = torch.zeros((), dtype=self.dtype, device=self.device)

        for batch in batches():
            values = self._tensor(batch)
            original_variance += values.flatten(1).square().sum(dim=1).sum()

        previous_variance = -1.0
        self.iterations = 0

        for iteration in range(1, self.max_iter + 1):
            for mode in range(3):
                others = [index for index in range(3) if index != mode]
                covariance = torch.zeros((shape[mode], shape[mode]), dtype=self.dtype, device=self.device)

                for batch in batches():
                    for volume in self._tensor(batch):
                        projected = volume

                        for other in others:
                            projected = self._mode_product(projected, bases[other].T, other)

                        unfolded = self._unfold(projected, mode)
                        covariance += unfolded @ unfolded.T

                eigenvalues, eigenvectors = torch.linalg.eigh(covariance)
                order = torch.argsort(eigenvalues, descending=True)
                order = order[: self.ranks[mode]]
                bases[mode] = eigenvectors[:, order]
                eigenvalue_sets[mode] = eigenvalues[order]

            preserved_variance = torch.zeros_like(original_variance)

            for batch in batches():
                for volume in self._tensor(batch):
                    projected = volume

                    for mode in range(3):
                        projected = self._mode_product(projected, bases[mode].T, mode)

                    preserved_variance += projected.square().sum()

            self.iterations = iteration

            if abs(preserved_variance.item() - previous_variance) < self.tol:
                break

            previous_variance = preserved_variance.item()

        self.bases = bases
        self.eigenvalues = eigenvalue_sets
        self.orig_shape = shape
        self.initialized = True
        return self

    def transform_batches(self, batches: Iterable[torch.Tensor] | Callable[[], Iterable[torch.Tensor]]) -> Iterator[torch.Tensor]:
        if not self.initialized:
            raise RuntimeError('Model should be fitted first')

        if self.projection_mode == 'axial':
            modes = (0,)
        elif self.projection_mode == 'frontal':
            modes = (1,)
        elif self.projection_mode == 'sagital':
            modes = (2,)
        else:
            modes = (0, 1, 2)

        source = batches() if callable(batches) else batches

        for batch in source:
            values = self._tensor(batch)
            if values.ndim != 4 or tuple(values.shape[1:]) != self.orig_shape:
                raise ValueError('Transform batches do not match fitted volume shape')

            transformed = []
            for volume in values:
                projected = volume

                for mode in modes:
                    projected = self._mode_product(projected, self.bases[mode].T, mode)

                transformed.append(projected)
            yield torch.stack(transformed)
