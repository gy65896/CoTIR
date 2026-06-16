"""
CoTIR dataset: WebDataset tar shards (per-degradation folders under a data_new-style root).
Each sample provides .hq.png, .lq.png, and .meta.json.
"""

import os
import random
import json
import io
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
import re

import numpy as np
from PIL import Image

import torch
from torch.utils.data import DataLoader, IterableDataset
import webdataset as wds

_MODULE_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _MODULE_DIR.parent
_POSSIBLE_PROMPT_ROOTS = [
    _PROJECT_ROOT / "CoTIR-Bench" / "prompts",
    _MODULE_DIR / "data_new",
]
for _path_candidate in _POSSIBLE_PROMPT_ROOTS:
    if _path_candidate.exists():
        _DEFAULT_PROMPT_ROOT = _path_candidate
        break
else:
    _DEFAULT_PROMPT_ROOT = _POSSIBLE_PROMPT_ROOTS[0]

_SPECIAL_PROMPT_CANDIDATES = [
    _DEFAULT_PROMPT_ROOT / "spe_prompt",
    _DEFAULT_PROMPT_ROOT / "prompts",
    _DEFAULT_PROMPT_ROOT,
]
for _special_candidate in _SPECIAL_PROMPT_CANDIDATES:
    if _special_candidate.exists():
        _DEFAULT_SPECIAL_PROMPT_ROOT = _special_candidate
        break
else:
    _DEFAULT_SPECIAL_PROMPT_ROOT = _DEFAULT_PROMPT_ROOT

_JPEG_PROMPT_ROOT = _DEFAULT_SPECIAL_PROMPT_ROOT
_jpeg_prompt_cache: Dict[str, List[str]] = {}

class PatchSizeController:
    """Global patch size controller for progressive training"""
    def __init__(self, patch_size=512):
        self.patch_size = patch_size
        self.sample_count = 0  # Track total samples processed
    
    def set_patch_size(self, patch_size):
        """Update patch size for progressive training"""
        self.patch_size = patch_size
        # Patch size updates are frequent; avoid noisy logging.
    
    def increment_count(self):
        """Increment sample count"""
        self.sample_count += 1
    
    def reset_count(self):
        """Reset sample count"""
        self.sample_count = 0
    
    def get_count(self):
        """Get current sample count"""
        return self.sample_count


# Global patch size controller
_patch_size_controller = PatchSizeController()


class PromptRepository:
    """Lazy loader for generic and per-degradation prompts."""

    def __init__(self):
        self._base_dir: Optional[Path] = None
        self._generic_prompts: Optional[List[str]] = None
        self._special_prompts: Dict[str, List[str]] = {}
        self._special_dir: Optional[Path] = None

    def configure(self, preferred_root: Optional[str]) -> None:
        candidate_values: List[str] = []
        if preferred_root:
            candidate_values.append(preferred_root)
        if _DEFAULT_PROMPT_ROOT.exists():
            candidate_values.append(str(_DEFAULT_PROMPT_ROOT))

        resolved_candidates: List[Path] = []
        for value in candidate_values:
            detected = _detect_prompt_root(value)
            if detected:
                resolved_candidates.append(Path(detected))
        if not resolved_candidates and candidate_values:
            fallback = Path(candidate_values[0])
            if fallback.exists():
                resolved_candidates.append(fallback)

        chosen: Optional[Path] = None
        for path in resolved_candidates:
            if path is None:
                continue
            if path.is_dir() and (path / "gen_prompt.txt").exists():
                chosen = path
                break
        if chosen is None and resolved_candidates:
            chosen = resolved_candidates[0]

        if chosen != self._base_dir:
            self._base_dir = chosen
            self._generic_prompts = None
            self._special_prompts = {}
            self._special_dir = self._discover_special_dir()
            _jpeg_prompt_cache.clear()

    def _read_lines(self, file_path: Path) -> List[str]:
        try:
            content = file_path.read_text(encoding="utf-8")
        except FileNotFoundError:
            return []
        return [line.strip() for line in content.splitlines() if line.strip()]

    def _discover_special_dir(self) -> Optional[Path]:
        if self._base_dir is None:
            return None
        base_dir = self._base_dir
        search_order: List[Path] = []

        # Prefer explicit subdirectories over the base folder itself.
        sub_candidates = [
            base_dir / "spe_prompt",
            base_dir / "prompts",
        ]
        for candidate in sub_candidates:
            if candidate not in search_order:
                search_order.append(candidate)

        if base_dir.name in ("prompts", "spe_prompt"):
            search_order.append(base_dir)

        # As a final fallback, look at the parent directory for standard layout.
        parent = base_dir.parent
        if parent and parent != base_dir:
            parent_sub = [
                parent / "spe_prompt",
                parent / "prompts",
            ]
            for candidate in parent_sub:
                if candidate not in search_order:
                    search_order.append(candidate)

        for candidate in search_order:
            if candidate and candidate.is_dir():
                return candidate
        return None

    def special_prompt_dir(self) -> Optional[Path]:
        return self._special_dir

    def _ensure_generic(self) -> List[str]:
        if self._generic_prompts is None:
            prompts: List[str] = []
            if self._base_dir is not None:
                prompts = self._read_lines(self._base_dir / "gen_prompt.txt")
            self._generic_prompts = prompts
        return self._generic_prompts

    def sample_generic(self) -> str:
        prompts = self._ensure_generic()
        return random.choice(prompts) if prompts else ""

    def _special_key(self, deg_label: Optional[str]) -> str:
        if not deg_label:
            return "unknown"
        return normalize_deg(deg_label)

    def _ensure_special(self, deg_label: Optional[str]) -> List[str]:
        key = self._special_key(deg_label)
        if key not in self._special_prompts:
            prompts: List[str] = []
            special_dir = self._special_dir
            if special_dir is None and self._base_dir is not None:
                fallback_dir = self._base_dir / "prompts"
                if fallback_dir.exists():
                    special_dir = fallback_dir

            if special_dir is not None:
                prompt_file = special_dir / f"{key}.txt"
                if prompt_file.exists():
                    prompts = self._read_lines(prompt_file)
                    if not prompts and key != "unknown":
                        raise ValueError(f"Special prompt file {prompt_file} is empty for {key}")
                elif key != "unknown":
                    raise FileNotFoundError(
                        f"Special prompt file not found for degradation '{key}': {prompt_file}"
                    )
            elif key != "unknown":
                raise FileNotFoundError(
                    f"No prompt root configured; missing special prompts for degradation '{key}'"
                )
            self._special_prompts[key] = prompts
        return self._special_prompts[key]

    def sample_special(self, deg_label: Optional[str]) -> Optional[str]:
        prompts = self._ensure_special(deg_label)
        if prompts:
            return random.choice(prompts)
        key = self._special_key(deg_label)
        if key and key != "unknown":
            desc = key.replace("_", " ").replace("-", " ")
            return f"Please correct {desc} issues."
        return None

    def require_special(self, deg_label: Optional[str]) -> None:
        key = self._special_key(deg_label)
        if key == "unknown":
            return
        self._ensure_special(deg_label)


_prompt_repository = PromptRepository()


def _weighted_prompt_pick(options: Sequence[Tuple[Optional[str], float]]) -> str:
    """Select a prompt string using weighted probabilities, skipping empty entries."""
    filtered: List[Tuple[str, float]] = [
        (text, weight) for text, weight in options if text and weight > 0
    ]
    if not filtered:
        return ""
    total_weight = sum(weight for _, weight in filtered)
    if total_weight <= 0:
        return filtered[-1][0]
    pick = random.random() * total_weight
    cumulative = 0.0
    for text, weight in filtered:
        cumulative += weight
        if pick <= cumulative:
            return text
    return filtered[-1][0]


def _load_jpeg_prompt_lines(jpeg_type: str) -> List[str]:
    """Load and cache prompts tied to a jpeg_type value."""
    if jpeg_type in _jpeg_prompt_cache:
        return _jpeg_prompt_cache[jpeg_type]
    search_dirs: List[Path] = []
    repo_special_dir = _prompt_repository.special_prompt_dir()
    if repo_special_dir is not None:
        search_dirs.append(repo_special_dir)
    if _JPEG_PROMPT_ROOT is not None and _JPEG_PROMPT_ROOT not in search_dirs:
        search_dirs.append(_JPEG_PROMPT_ROOT)

    for directory in search_dirs:
        file_path = directory / f"{jpeg_type}.txt"
        if not file_path.exists():
            continue
        content = file_path.read_text(encoding="utf-8")
        prompts = [line.strip() for line in content.splitlines() if line.strip()]
        _jpeg_prompt_cache[jpeg_type] = prompts
        return prompts

    _jpeg_prompt_cache[jpeg_type] = []
    return []


def _sample_jpeg_type_prompt(jpeg_type: Optional[str]) -> Optional[str]:
    """Sample a prompt for the provided jpeg_type, if available."""
    if not jpeg_type:
        return None
    prompts = _load_jpeg_prompt_lines(str(jpeg_type))
    if not prompts:
        return None
    return random.choice(prompts)


def _normalize_prompt_dir(path: Path, max_parent_checks: int = 3) -> Optional[Path]:
    if path is None:
        return None
    candidate = path.expanduser()
    if not candidate.exists():
        return None

    search_paths: List[Path] = []
    seen: set = set()

    def enqueue(target: Optional[Path]) -> None:
        if target is None:
            return
        if target in seen:
            return
        seen.add(target)
        search_paths.append(target)

    enqueue(candidate)
    if candidate.is_dir():
        enqueue(candidate / "prompts")
        enqueue(candidate / "spe_prompt")

    ancestor_checks = 0
    for ancestor in candidate.parents:
        if ancestor_checks >= max_parent_checks:
            break
        enqueue(ancestor)
        enqueue(ancestor / "prompts")
        ancestor_checks += 1

    for option in search_paths:
        if option.is_dir():
            gen_file = option / "gen_prompt.txt"
            if gen_file.exists():
                return option
    return None


def _detect_prompt_root(data_dir: Optional[str]) -> Optional[str]:
    if not data_dir:
        return None
    normalized = _normalize_prompt_dir(Path(data_dir))
    if normalized is None:
        return None
    return str(normalized)


def _resolve_balanced_root_and_prompts(data_dir: str) -> Tuple[str, Optional[str]]:
    """
    Normalize balanced_new data directory so it can point either to the dataset root
    or directly to the image folder, while also surfacing a co-located prompt folder.
    """
    if not data_dir:
        return data_dir, None

    base_path = Path(data_dir).expanduser()
    if not base_path.exists():
        return data_dir, None

    def _prompt_dir(parent: Path) -> Optional[str]:
        prompt_path = parent / "prompts"
        if prompt_path.is_dir():
            return str(prompt_path)
        return None

    if base_path.is_dir() and base_path.name == "image":
        prompt_dir = _prompt_dir(base_path.parent)
        return str(base_path), prompt_dir

    image_dir = base_path / "image"
    if image_dir.is_dir():
        prompt_dir = _prompt_dir(base_path)
        return str(image_dir), prompt_dir

    return str(base_path), None


def _make_tuple_to_sample_fn(include_deg_type: bool, deg_label_override: Optional[str] = None):
    """
    Factory returning a tuple->sample mapper with optional deg label override.
    """

    def tuple_to_sample(sample_tuple):
        sample_dict = {
            "hq.png": sample_tuple[0],
            "lq.png": sample_tuple[1],
            "meta.json": sample_tuple[2],
            "__key__": sample_tuple[3],
        }
        if include_deg_type:
            if deg_label_override is not None:
                sample_dict["deg_type"] = deg_label_override
            elif len(sample_tuple) > 4:
                sample_dict["deg_type"] = _infer_deg_type_from_url(sample_tuple[4])
        return decode_sample(sample_dict, include_deg_type=include_deg_type)

    return tuple_to_sample


class StringList(list):
    """Wrapper for list of strings that prevents Accelerate from trying to concatenate
    
    Inherits from list so it can be used with tokenizers, but Accelerate won't try to concatenate it
    """
    pass


def normalize_deg(value: Optional[str]) -> str:
    if not value:
        return "unknown"
    normalized = value.strip().lower()
    normalized = re.sub(r"[^a-z0-9._-]+", "_", normalized)
    return normalized or "unknown"


def _list_degradation_shards(root_dir: str, target_types: Optional[Sequence[str]] = None) -> Dict[str, List[str]]:
    """Return mapping {deg_type: [tar paths]} for the balanced data_new layout."""
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Balanced data directory not found: {root_dir}")

    if target_types:
        type_names = [t for t in target_types if os.path.isdir(os.path.join(root_dir, t))]
    else:
        type_names = sorted(
            entry
            for entry in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, entry))
        )

    import glob as glob_module

    shards_map: Dict[str, List[str]] = {}
    for deg_type in type_names:
        tar_pattern = os.path.join(root_dir, deg_type, "*.tar")
        tar_files = sorted(glob_module.glob(tar_pattern))
        if tar_files:
            shards_map[deg_type] = tar_files

    return shards_map


def _infer_deg_type_from_url(url_value) -> str:
    """Best-effort extraction of degradation type from WebDataset __url__ value."""
    if not url_value:
        return "unknown"
    if isinstance(url_value, bytes):
        url = url_value.decode("utf-8", errors="ignore")
    else:
        url = str(url_value)

    # Remove common prefixes
    for prefix in ("pipe:", "file:", "http://", "https://"):
        if url.startswith(prefix):
            url = url[len(prefix):]
    if "::" in url:
        url = url.split("::", 1)[-1]
    url = url.split("?", 1)[0]
    url = url.split("@", 1)[0]
    url = url.strip()
    if not url:
        return "unknown"
    type_dir = os.path.basename(os.path.dirname(url))
    return type_dir or "unknown"


def _count_meta_in_tar(tar_path: str) -> int:
    """Count .meta.json files inside a tar shard."""
    import tarfile

    try:
        with tarfile.open(tar_path, "r:*") as tar:
            return sum(1 for member in tar.getmembers() if member.name.endswith(".meta.json"))
    except Exception:
        return 0


def _estimate_balanced_sample_counts(shards_map: Dict[str, List[str]]) -> Tuple[int, Dict[str, int]]:
    """Estimate total and per-type sample counts for balanced layout."""
    per_type: Dict[str, int] = {}
    total = 0
    for deg, shards in shards_map.items():
        if not shards:
            continue
        per_shard = _count_meta_in_tar(shards[0])
        estimated = per_shard * len(shards)
        per_type[deg] = estimated
        total += estimated
    return total, per_type


def _choose_specific_prompt(
    questions: Sequence[str],
    special_prompt: Optional[str],
    deg_label: Optional[str] = None,
) -> str:
    question_weight = 0.3
    special_weight = 0.7
    if deg_label and "lost_color" in deg_label:
        question_weight = 0.8
        special_weight = 0.2
    question_text = questions[1] if len(questions) > 1 else (questions[0] if questions else None)
    return _weighted_prompt_pick(
        [
            (question_text, question_weight),
            (special_prompt, special_weight),
        ]
    )


def decode_sample(sample, count_samples=False, include_deg_type=False):
    """
    Decode a single sample from CoTIR Dataset
    Uses global patch size controller for progressive training
    
    Args:
        sample: dict with keys '__key__', 'hq.png', 'lq.png', 'meta.json'
        count_samples: If True, increment global sample counter
    
    Returns:
        (hq_img, lq_img, prompt, answer)
    """
    try:
        # Get current patch size from global controller
        patch_size = _patch_size_controller.patch_size
        
        # Decode images
        hq_img = Image.open(io.BytesIO(sample['hq.png'])).convert('RGB')
        lq_img = Image.open(io.BytesIO(sample['lq.png'])).convert('RGB')
        
        # Decode metadata
        meta_dict = json.loads(sample['meta.json'].decode('utf-8'))
        
        # Extract prompt and answer
        cot = meta_dict['cot']
        questions = cot.get('questions') or []
        deg_label_raw = sample.get("deg_type") or meta_dict.get("deg") or meta_dict.get("degradation") or meta_dict.get("degradation_type")
        deg_label = normalize_deg(deg_label_raw) if deg_label_raw else "unknown"
        
        # Randomly select prompts from external files (with fallbacks) using configured repository
        generic_prompt = _prompt_repository.sample_generic()
        first_question = questions[0] if questions else None
        questions_gen = _weighted_prompt_pick(
            [
                (generic_prompt, 0.8),
                (first_question, 0.2),
            ]
        )
        special_prompt = _prompt_repository.sample_special(deg_label)
        jpeg_type_label = meta_dict.get("jpeg_type")
        if jpeg_type_label and (
            ("jpeg_compression" in deg_label) or ("jpeg compression" in (deg_label_raw or "").lower())
        ):
            jpeg_prompt = _sample_jpeg_type_prompt(jpeg_type_label)
            if jpeg_prompt:
                special_prompt = jpeg_prompt
        questions_spe = _choose_specific_prompt(questions, special_prompt, deg_label=deg_label)
        # Parse the "answer" field into scene, degradation, and enhancement plan using known prefixes
        # Example answer format:
        # 'scene description: ...degradation description: ...enhancement plan: ...answer: ...'
        answer_text = cot['answers']
        def extract_between(text, start_key, end_key):
            start_idx = text.find(start_key)
            if start_idx == -1:
                return ""
            start_idx += len(start_key)
            if end_key is not None:
                end_idx = text.find(end_key, start_idx)
                if end_idx == -1:
                    return text[start_idx:].strip()
                return text[start_idx:end_idx].strip()
            return text[start_idx:].strip()

        answer_s = extract_between(
            answer_text,
            "scene description:",
            "degradation description:"
        )
        answer_d = extract_between(
            answer_text,
            "degradation description:",
            "enhancement plan:"
        )
        answer_p = extract_between(
            answer_text,
            "enhancement plan:",
            "answer:"
        )
        # answer = cot['answers']
        
        # Convert to numpy arrays
        hq_arr = np.array(hq_img)
        lq_arr = np.array(lq_img)
        
        # Convert to torch tensors and normalize to [-1, 1]
        hq_tensor = torch.from_numpy(hq_arr.astype(np.float32) / 127.5 - 1)
        hq_tensor = hq_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        lq_tensor = torch.from_numpy(lq_arr.astype(np.float32) / 127.5 - 1)
        lq_tensor = lq_tensor.permute(2, 0, 1)  # HWC -> CHW
        
        # Resize if patch_size is different from 512
        if patch_size != 512:
            hq_tensor = torch.nn.functional.interpolate(
                hq_tensor.unsqueeze(0), size=(patch_size, patch_size), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
            lq_tensor = torch.nn.functional.interpolate(
                lq_tensor.unsqueeze(0), size=(patch_size, patch_size), 
                mode='bilinear', align_corners=False
            ).squeeze(0)
        
        # Increment counter if requested (for counting total samples)
        if count_samples:
            _patch_size_controller.increment_count()

        outputs = (
            hq_tensor,
            lq_tensor,
            questions_gen,
            questions_spe,
            answer_s,
            answer_d,
            answer_p,
        )

        if include_deg_type:
            return outputs + (deg_label,)
        
        return outputs
    
    except Exception as e:
        raise RuntimeError(f"Error decoding sample {sample.get('__key__', 'unknown')}: {e}") from e


def data_loader(
    train_batch_size,
    num_workers,
    data_dir,
    test=False,
    patch_size=512,
    shuffle_buffer=500,
    print_sample_count=False,
    balanced_target_types: Optional[Sequence[str]] = None,
    balanced_rounds: Optional[int] = None,
    balanced_shuffle: bool = True,
    balanced_seed: Optional[int] = None,
    return_deg_type: Optional[bool] = None,
):
    """
    WebDataset loader: `data_new` layout (per-degradation folders with ``*.tar`` shards).

    Args:
        data_dir: Dataset root (or ``.../image``); see ``_resolve_balanced_root_and_prompts``.
        return_deg_type: If False, omit ``deg_type`` from batches. If omitted, labels are included.
    """
    if not data_dir:
        raise ValueError("data_dir must be provided")
    data_dir = os.fspath(data_dir)
    include_deg_type = True if return_deg_type is None else bool(return_deg_type)

    data_root, prompt_root_candidate = _resolve_balanced_root_and_prompts(data_dir)
    shards_map = _list_degradation_shards(data_root, balanced_target_types)
    if not shards_map:
        raise FileNotFoundError(f"No degradation tar files found under {data_root}")

    if print_sample_count:
        total_samples, _ = _estimate_balanced_sample_counts(shards_map)
        print(f"[CoTIR] Estimated total samples (balanced): {total_samples:,}")
    
    _patch_size_controller.set_patch_size(patch_size)
    prompt_candidates: List[str] = []
    if prompt_root_candidate:
        prompt_candidates.append(prompt_root_candidate)
    prompt_candidates.append(os.path.abspath(data_root))
    if data_root != data_dir:
        prompt_candidates.append(os.path.abspath(data_dir))

    resolved_prompt_root: Optional[str] = None
    for candidate in prompt_candidates:
        detected = _detect_prompt_root(candidate)
        if detected:
            resolved_prompt_root = detected
            break

    _prompt_repository.configure(resolved_prompt_root)

    class DatasetWithPatchSize(IterableDataset):
        """Wrapper providing set_patch_size interface for CoTIR Dataset"""
        def __init__(self, wds_dataset):
            super().__init__()
            self.wds_dataset = wds_dataset
        
        def set_patch_size(self, size):
            _patch_size_controller.set_patch_size(size)
        
        def __iter__(self):
            return iter(self.wds_dataset)

    class RoundRobinBalancedDataset(IterableDataset):
        """Iterate over per-type datasets in strict round-robin order."""

        def __init__(
            self,
            per_type_datasets: Dict[str, IterableDataset],
            type_names: Sequence[str],
            shuffle_types: bool = True,
            seed: Optional[int] = None,
            round_limit: Optional[int] = None,
        ):
            super().__init__()
            if not per_type_datasets or not type_names:
                raise ValueError("RoundRobinBalancedDataset requires at least one degradation type")
            self.per_type_datasets = per_type_datasets
            self.type_names = list(type_names)
            self.shuffle_types = shuffle_types
            self.seed = seed
            self.round_limit = round_limit if round_limit and round_limit > 0 else None

        def __iter__(self):
            worker_info = torch.utils.data.get_worker_info()
            worker_seed = worker_info.seed if worker_info is not None else 0
            base_seed = self.seed if self.seed is not None else random.randint(0, 2**31 - 1)
            rng = random.Random(base_seed + worker_seed)

            per_type_iters = {deg: iter(ds) for deg, ds in self.per_type_datasets.items()}
            completed_cycles = 0

            while True:
                type_order = self.type_names[:]
                if self.shuffle_types and len(type_order) > 1:
                    rng.shuffle(type_order)

                produced = False
                for deg in type_order:
                    sample = self._next_sample(deg, per_type_iters)
                    if sample is None:
                        continue
                    produced = True
                    yield sample

                if not produced:
                    break

                completed_cycles += 1
                if self.round_limit is not None and completed_cycles >= self.round_limit:
                    break

        def _next_sample(self, deg: str, per_type_iters: Dict[str, Iterable]):
            dataset = self.per_type_datasets[deg]
            iterator = per_type_iters.get(deg)
            attempts = 0

            while True:
                if iterator is None:
                    iterator = iter(dataset)
                    per_type_iters[deg] = iterator
                try:
                    return next(iterator)
                except StopIteration:
                    iterator = iter(dataset)
                    per_type_iters[deg] = iterator
                    attempts += 1
                    if attempts > 8:
                        return None
                except Exception:
                    return None

    tuple_fields = ['hq.png', 'lq.png', 'meta.json', '__key__']
    if include_deg_type:
        tuple_fields.append('__url__')

    per_type_datasets: Dict[str, IterableDataset] = {}
    balanced_types = sorted(shards_map.keys())
    if test and balanced_types:
        balanced_types = balanced_types[:1]

    for deg_type in balanced_types:
        tar_files = shards_map.get(deg_type, [])
        if test:
            tar_files = tar_files[:1]
        if not tar_files:
            continue

        _prompt_repository.require_special(deg_type)

        type_shardshuffle = 1000 if (len(tar_files) > 1 and not test) else False
        per_type_dataset = (
            wds.WebDataset(
                tar_files,
                shardshuffle=type_shardshuffle,
                nodesplitter=wds.split_by_node,
                resampled=not test,
            )
            .shuffle(shuffle_buffer)
            .to_tuple(*tuple_fields)
            .map(
                _make_tuple_to_sample_fn(
                    include_deg_type,
                    deg_label_override=deg_type if include_deg_type else None,
                )
            )
            .select(lambda x: x is not None)
        )
        per_type_datasets[deg_type] = per_type_dataset

    if not per_type_datasets:
        raise RuntimeError("Failed to initialize balanced datasets")

    type_names = sorted(per_type_datasets.keys())
    round_limit = balanced_rounds if balanced_rounds and balanced_rounds > 0 else None
    if test and round_limit is None:
        round_limit = 1
    dataset_core = RoundRobinBalancedDataset(
        per_type_datasets=per_type_datasets,
        type_names=type_names,
        shuffle_types=balanced_shuffle,
        seed=balanced_seed,
        round_limit=round_limit,
    )

    def collate_fn(batch):
        """Collate function that handles tensors and strings separately."""
        hq_imgs = torch.stack([item[0] for item in batch])
        lq_imgs = torch.stack([item[1] for item in batch])
        questions_gen = StringList([item[2] for item in batch])
        questions_spe = StringList([item[3] for item in batch])
        answers_s = StringList([item[4] for item in batch])
        answers_d = StringList([item[5] for item in batch])
        answers_p = StringList([item[6] for item in batch])
        if include_deg_type:
            deg_types = StringList([item[7] for item in batch])
            return (hq_imgs, lq_imgs, questions_gen, questions_spe, answers_s, answers_d, answers_p, deg_types)
        return (hq_imgs, lq_imgs, questions_gen, questions_spe, answers_s, answers_d, answers_p)
    
    dataset = DatasetWithPatchSize(dataset_core)
    
    loader = DataLoader(
        dataset,
        batch_size=train_batch_size,
        num_workers=num_workers,
        collate_fn=collate_fn,
        prefetch_factor=1 if num_workers > 0 else None,
        persistent_workers=False,
        drop_last=True,
        pin_memory=False,
    )
    
    return loader


if __name__ == "__main__":
    # Test loading
    print("Testing CoTIR Dataset loader...")
    loader = data_loader(
        train_batch_size=4,
        num_workers=0,
        data_dir='/data1/guoyu_data/CoTIR/CoTIR-Bench',
        test=False,
        patch_size=128
    )
    
    print("Loading first batch...")
    for batch_idx, batch in enumerate(loader):
        if len(batch) == 8:
            hq_imgs, lq_imgs, questions_gen, questions_spe, answers_s, answers_d, answers_p, deg_types = batch
        else:
            hq_imgs, lq_imgs, questions_gen, questions_spe, answers_s, answers_d, answers_p = batch
            deg_types = None
        print(f"\nBatch {batch_idx}:")
        print(f"  HQ shape: {hq_imgs.shape}")
        print(f"  LQ shape: {lq_imgs.shape}")
        print(f"  Questions gen: {questions_gen}")
        print(f"  Questions spe: {questions_spe}")
        print(f"  Answers s: {answers_s}")
        print(f"  Answers d: {answers_d}")
        print(f"  Answers p: {answers_p}")
        if deg_types is not None:
            print(f"  Deg types: {deg_types}")

        if batch_idx >= 2:  # Test 3 batches
            break
    
    print("\n✓ Test passed!")

