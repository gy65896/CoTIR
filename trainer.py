
from accelerate.logging import get_logger
from accelerate.utils import ProjectConfiguration
from accelerate import Accelerator
from diffusers.optimization import get_scheduler
import datasets, transformers, diffusers
import os, logging, torch, time
import torch.multiprocessing as mp
import glob
import gc
from pathlib import Path
from omegaconf import OmegaConf
from cotir.model import load_model
from data import data_loader
from tqdm import tqdm
from cotir.flux.util import configs as FLUX_MODEL_CONFIGS
from cotir.flux2.util import FLUX2_MODEL_INFO
from cotir.util import (
    load_infer_backends_by_model,
    train_one_step,
    val_one_step,
    visualize_results,
    save_checkpoint,
)


def get_current_step(global_step, training_config: dict):
    """Get current training stage based on global_step."""
    iterations = training_config['iterations']
    patch_sizes = training_config['patch_sizes']
    batch_sizes = training_config['batch_sizes']
    
    # Find current stage index based on iterations
    stage_index = 0
    for i, milestone in enumerate(iterations):
        if global_step < milestone:
            stage_index = i
            break
    else:
        # If beyond all milestones, use last stage
        stage_index = len(patch_sizes) - 1
    
    return batch_sizes[stage_index], patch_sizes[stage_index]

class Trainer:
    def __init__(self, config):
        self.config = config
        self.device = config.device
        self.mixed_precision = config.mixed_precision
        self.output_dir = config.data.output_dir
        self.total_iterations = config.training.iterations[-1]
        self._initialize_runtime()
        self._build_training_components()
        self._prepare_with_accelerator()
        self._restore_checkpoint_states()
        
        
    def _initialize_runtime(self):
        self.init_config()
        self._configure_multiprocessing()

    def _build_training_components(self):
        self.get_models()
        self.trainable_parameters()
        self.get_optimizer()
        self.get_lagrange_multiplier()

    def _configure_multiprocessing(self):
        try:
            mp.set_sharing_strategy("file_system")
            self.logger.info("Using torch multiprocessing sharing strategy: file_system")
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:512'
            os.environ['TORCH_SHOW_CPP_STACKTRACES'] = '1'
        except RuntimeError as e:
            self.logger.warning(f"Failed to set sharing strategy: {e}")

    def _prepare_with_accelerator(self):
        self.optimizer, self.model, self.lr_scheduler = self.accelerator.prepare(
            self.optimizer, self.model, self.lr_scheduler
        )

    def _restore_checkpoint_states(self):
        if not getattr(self, 'optimizer_path', None):
            return

        self._load_state_file(
            self.optimizer_path,
            "optimizer state",
            lambda state: self.optimizer.load_state_dict(state),
            "✓ Optimizer state loaded successfully",
        )

        lr_scheduler_path = self.optimizer_path.replace('optimizer.bin', 'lr_scheduler.bin')
        self._load_state_file(
            lr_scheduler_path,
            "lr_scheduler state",
            lambda state: self.lr_scheduler.load_state_dict(state),
            "✓ LR scheduler state loaded successfully",
        )

        if not getattr(self, 'use_adaptive_weight', False):
            return

        lambda_params = getattr(self, 'lambda_params', None)
        if lambda_params is not None and len(lambda_params) > 0:
            lambda_param_path = self.optimizer_path.replace('optimizer.bin', 'lambda_param.bin')

            def load_lambda_params(state_dict):
                with torch.no_grad():
                    if isinstance(state_dict, dict):
                        items = state_dict.items()
                    else:
                        # Backwards compatibility: single tensor stored
                        items = zip(lambda_params.keys(), [state_dict])
                    for key, value in items:
                        if key in lambda_params:
                            lambda_params[key].data.copy_(value.to(lambda_params[key].device))

            self._load_state_file(
                lambda_param_path,
                "lambda parameters",
                load_lambda_params,
                "✓ Lambda parameters loaded successfully",
            )

        lambda_optimizer = getattr(self, 'lambda_optimizer', None)
        if lambda_optimizer is not None:
            lambda_optimizer_path = self.optimizer_path.replace('optimizer.bin', 'lambda_optimizer.bin')
            self._load_state_file(
                lambda_optimizer_path,
                "lambda_optimizer state",
                lambda state: lambda_optimizer.load_state_dict(state),
                "✓ Lambda_optimizer state loaded successfully",
            )

    def _load_state_file(self, path, description, apply_fn, success_message):
        if not path or not os.path.exists(path):
            return

        try:
            self.logger.info(f"Loading {description} from {path}")
            state = torch.load(path, map_location='cpu')
            apply_fn(state)
        except Exception as error:  # noqa: B902
            self.logger.warning(f"Failed to load {description}: {error}")
        else:
            self.logger.info(success_message)

    @staticmethod
    def _extract_state_dict(obj):
        if isinstance(obj, dict):
            for key in ("state_dict", "model_state_dict", "model", "ema", "weights"):
                value = obj.get(key)
                if isinstance(value, dict):
                    return value
            return obj
        if hasattr(obj, "state_dict") and callable(obj.state_dict):
            return obj.state_dict()
        raise TypeError(f"Unsupported checkpoint object type: {type(obj)!r}")

    @staticmethod
    def _strip_module_prefix(sd: dict):
        return {k.replace("module.", "", 1) if k.startswith("module.") else k: v for k, v in sd.items()}

    def _load_checkpoint_into_model(self, model, checkpoint_path: str, description: str):
        if not checkpoint_path or not os.path.exists(checkpoint_path):
            return
        self.logger.info(f"Loading {description} from {checkpoint_path}")
        state = torch.load(checkpoint_path, map_location="cpu")
        state = self._strip_module_prefix(self._extract_state_dict(state))
        missing, unexpected = model.load_state_dict(state, strict=False)
        if missing:
            self.logger.info(f"{description}: missing keys={len(missing)}")
        if unexpected:
            self.logger.info(f"{description}: unexpected keys={len(unexpected)}")

    def init_config(self):
        logger = get_logger(__name__, log_level="INFO")
        logging.basicConfig(
            format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
            datefmt="%m/%d/%Y %H:%M:%S",
            level=logging.INFO,
        )

        self.logging_dir = os.path.join(
            self.output_dir, 
            self.config.data.logging_dir
        )

        accelerator_project_config = ProjectConfiguration(
            project_dir=self.output_dir, 
            logging_dir=self.logging_dir
        )

        # Configure DDP to handle unused parameters
        from accelerate import DistributedDataParallelKwargs
        ddp_kwargs = DistributedDataParallelKwargs()

        accelerator = Accelerator(
            gradient_accumulation_steps=self.config.training.gradient_accumulation_steps,
            mixed_precision=self.config.mixed_precision,
            log_with=self.config.report_to,
            project_config=accelerator_project_config,
            kwargs_handlers=[ddp_kwargs],
        )
        weight_dtype = torch.float16 if accelerator.mixed_precision == "fp16" else torch.bfloat16 if accelerator.mixed_precision == "bf16" else torch.float32
        self.weight_dtype = weight_dtype

        logger.info(accelerator.state, main_process_only=False)
        if accelerator.is_local_main_process:
            datasets.utils.logging.set_verbosity_warning()
            transformers.utils.logging.set_verbosity_warning()
            diffusers.utils.logging.set_verbosity_info()
        else:
            datasets.utils.logging.set_verbosity_error()
            transformers.utils.logging.set_verbosity_error()
            diffusers.utils.logging.set_verbosity_error()

        if accelerator.is_main_process:
            if self.output_dir is not None:
                os.makedirs(self.output_dir, exist_ok=True)

        self.accelerator = accelerator
        self.logger = logger
    
    def get_models(self):   
        print("Loading models...")
        base_model_name = str(self.config.model.base_model_name).lower()
        is_flux2 = base_model_name.startswith("flux.2")
        model_base_path_cfg = getattr(self.config.model, "base_model_path", None)
        if model_base_path_cfg is None:
            raise ValueError("Training config must set model.base_model_path for local initialization.")
        model_base_path = Path(str(model_base_path_cfg)).expanduser().resolve()

        if is_flux2:
            if base_model_name not in FLUX2_MODEL_INFO:
                raise KeyError(f"Unsupported flux2 base model: {self.config.model.base_model_name}")
            expected_name = FLUX2_MODEL_INFO[base_model_name]["filename"]
        else:
            if base_model_name not in FLUX_MODEL_CONFIGS:
                raise KeyError(f"Unsupported kontext base model: {self.config.model.base_model_name}")
            expected_name = FLUX_MODEL_CONFIGS[base_model_name].repo_flow

        base_ckpt_path = model_base_path if model_base_path.is_file() else (model_base_path / expected_name)
        if not base_ckpt_path.exists():
            raise FileNotFoundError(f"Missing local backbone checkpoint: {base_ckpt_path}")
        base_checkpoint_path = str(base_ckpt_path)
        self.logger.info(f"Using local backbone checkpoint: {base_checkpoint_path}")

        cfg_for_backends = OmegaConf.create(
            {
                "model": {"base_model_name": self.config.model.base_model_name},
                "inference": {"base_model_path": str(model_base_path)},
            }
        )
        t5, clip, ae = load_infer_backends_by_model(
            cfg=cfg_for_backends,
            device=self.accelerator.device,
            base_ckpt=base_ckpt_path,
        )
        
        # Initialize use_gen from config
        self.use_gen = getattr(self.config.training, "use_gen", None)
        
        # Handle resume from checkpoint
        latest = None
        checkpoint_dir = None
        optimizer_path = None
        if self.config.resume and self.config.resume == "latest":
            # Find latest checkpoint
            if os.path.exists(self.output_dir):
                checkpoints = [d for d in os.listdir(self.output_dir) if d.startswith("checkpoint")]
                if checkpoints:
                    latest = sorted(checkpoints, key=lambda x: int(x.split("-")[1]))[-1]
                    checkpoint_dir = os.path.join(self.output_dir, latest)
                    optimizer_path = os.path.join(self.output_dir, latest, 'optimizer.bin')
        model = load_model(
            self.config.model,
            checkpoint_path=base_checkpoint_path,
            device="cpu",
            verbose=False,
            for_inference=False,
        )
        if checkpoint_dir:
            split_lora = os.path.join(checkpoint_dir, "lora.pt")
            split_cot = os.path.join(checkpoint_dir, "cot_adapter.pt")
            split_base = os.path.join(checkpoint_dir, "base.pt")
            has_any_split = os.path.exists(split_lora) or os.path.exists(split_cot) or os.path.exists(split_base)
            if not has_any_split:
                raise FileNotFoundError(
                    f"Resume checkpoint {checkpoint_dir} has no split weights "
                    f"(expected any of: base.pt, cot_adapter.pt, lora.pt)."
                )
            self._load_checkpoint_into_model(model, split_base, "base split checkpoint")
            self._load_checkpoint_into_model(model, split_cot, "cot_adapter split checkpoint")
            self._load_checkpoint_into_model(model, split_lora, "lora split checkpoint")
        
        self.optimizer_path = optimizer_path
        # Safely calculate global_step
        if latest:
            try:
                self.global_step = int(latest.split("-")[1]) + 1
            except (ValueError, IndexError, AttributeError) as e:
                self.logger.warning(f"Failed to parse checkpoint step from '{latest}': {e}. Starting from step 0.")
                self.global_step = 0
        else:
            self.global_step = 0
        batch_size, patch_size = get_current_step(self.global_step, self.config.training)

        self.batch_size = batch_size
        self.patch_size = patch_size
        # Don't convert model here, accelerator.prepare() will handle it
        self.model = model
        self.ae = ae
        self.t5 = t5
        self.clip = clip
    
    def trainable_parameters(self):
        trainable_parameters = []
        for name, param in self.model.named_parameters():
            if 'cot_adapter' in name or 'lora' in name:
                trainable_parameters.append(name)
                param.requires_grad_(True)
            else:
                param.requires_grad_(False)
        
        self.logger.info(f"Trainable parameters: {trainable_parameters}")
    
    def get_optimizer(self):
        optimizer_cls = torch.optim.AdamW
        optimizer = optimizer_cls(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.config.training.learning_rate,
            betas=(self.config.training.adam_beta1, self.config.training.adam_beta2),
            weight_decay=self.config.training.adam_weight_decay,
            eps=self.config.training.adam_epsilon,
        )
        
        self.optimizer = optimizer

        self.lr_scheduler = get_scheduler(
            self.config.training.lr_scheduler,
            optimizer=self.optimizer,
            num_warmup_steps=self.config.training.lr_warmup_steps * self.accelerator.num_processes,
            num_training_steps=self.total_iterations * self.accelerator.num_processes,
        )
    def get_lagrange_multiplier(self):
        """Initialize Lagrange multipliers for constrained optimization."""
        for key in ("s", "d", "p"):
            setattr(self, f"lambda_param_{key}", None)

        lambda_param_dict = torch.nn.ParameterDict(
            {
                "s": torch.nn.Parameter(torch.tensor(0.0), requires_grad=True),
                "d": torch.nn.Parameter(torch.tensor(0.0), requires_grad=True),
                "p": torch.nn.Parameter(torch.tensor(0.0), requires_grad=True),
            }
        )
        lambda_lr = getattr(self.config.training, "lambda_lr", 1e-4)
        lambda_optimizer = torch.optim.Adam(
            lambda_param_dict.parameters(),
            lr=lambda_lr,
            betas=(0.9, 0.999),
        )
        prepared_params, prepared_optimizer = self.accelerator.prepare(
            lambda_param_dict, lambda_optimizer
        )

        if hasattr(prepared_params, "module"):
            prepared_params = prepared_params.module

        self.lambda_params = prepared_params
        self.lambda_optimizer = prepared_optimizer
        self.use_adaptive_weight = True
        self.adaptive_lambda_keys = ["s", "d", "p"]

        for key in ("s", "d", "p"):
            setattr(self, f"lambda_param_{key}", self.lambda_params[key])

        self.logger.info("✓ Using Lagrange multipliers (primal-dual) for text constraints")
        self.logger.info(f"  lambda_lr: {lambda_lr}")

    def _current_txt_weight(self, key: str, fallback_value: float) -> float:
        if key in self.lambda_params:
            return self.lambda_params[key].item()
        return float(fallback_value)

    def cleanup_shared_memory(self):
        if not self.accelerator.is_main_process:
            return

        try:
            # Be conservative with torch_* files to avoid racing active workers.
            torch_shm_files = glob.glob('/dev/shm/torch_*')
            stale_torch_files = []
            total_size = 0
            for shm_file in torch_shm_files:
                try:
                    file_age = time.time() - os.path.getmtime(shm_file)
                    file_size = os.path.getsize(shm_file)
                    # Only purge very old torch shared-memory blobs (2 minutes+)
                    if file_age > 120:
                        total_size += file_size
                        os.remove(shm_file)
                        stale_torch_files.append(shm_file)
                except (OSError, PermissionError, FileNotFoundError):
                    continue

            if stale_torch_files:
                size_mb = total_size / (1024 * 1024)
                self.logger.info(
                    f"Cleaned {len(stale_torch_files)} stale torch shared-memory files (~{size_mb:.2f} MB)"
                )

            other_patterns = ['/dev/shm/pymp-*', '/dev/shm/pyshmem-*']
            for pattern in other_patterns:
                for other_file in glob.glob(pattern):
                    try:
                        file_age = time.time() - os.path.getmtime(other_file)
                        if file_age > 30:
                            os.remove(other_file)
                    except (OSError, PermissionError, FileNotFoundError):
                        continue

            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as exc:  # noqa: B902
            self.logger.warning(f"Failed to cleanup shared memory: {exc}")

    def load_data(self):
        data_cfg = self.config.data

        train_dataloader = data_loader(
            train_batch_size=self.batch_size // self.config.training.gradient_accumulation_steps,
            num_workers=data_cfg.num_workers,
            data_dir=data_cfg.data_dir,
            test=getattr(data_cfg, "test", False),
            patch_size=self.patch_size,
            shuffle_buffer=getattr(data_cfg, "shuffle_buffer", 500),
            print_sample_count=getattr(data_cfg, "print_sample_count", False),
            balanced_target_types=getattr(data_cfg, "balanced_target_types", None),
            balanced_rounds=getattr(data_cfg, "balanced_rounds", None),
            balanced_shuffle=getattr(data_cfg, "balanced_shuffle", True),
            balanced_seed=getattr(data_cfg, "balanced_seed", None),
            return_deg_type=getattr(data_cfg, "return_deg_type", False),
        )
        return iter(train_dataloader)

    def start_training(self):
        print("Starting training...")
        if self.accelerator.is_main_process:
            self.accelerator.init_trackers(self.config.project_name, {"test": None})

        progress_bar = tqdm(
            initial=self.global_step,
            total=self.total_iterations,
            desc='Training progress',
            disable=not self.accelerator.is_local_main_process
        )

        stage_boundaries = list(self.config.training.iterations)
        self.model.train()

        try:
            for stage_idx, stage_end in enumerate(stage_boundaries):
                if self.global_step >= self.total_iterations:
                    break
                if self.global_step >= stage_end:
                    continue

                stage_iterator = self._start_stage(stage_idx, stage_end)

                while self.global_step < stage_end and self.global_step < self.total_iterations:
                    train_loss = 0.0
                    train_img_loss = 0.0
                    train_txt_s_loss = 0.0
                    train_txt_d_loss = 0.0
                    train_txt_p_loss = 0.0

                    try:
                        batch = next(stage_iterator)
                    except StopIteration:
                        stage_iterator = self.load_data()
                        batch = next(stage_iterator)

                    # Compute use_gen_prob outside accumulate block for logging
                    current_use_gen_prob = self._compute_use_gen_prob()

                    with self.accelerator.accumulate(self.model):
                        delta_s = self.config.training.delta_s
                        delta_d = self.config.training.delta_d
                        delta_p = self.config.training.delta_p
                        dict_input = {
                            'model': self.model, 'batch': batch,
                            'lambda_s': self.lambda_params["s"],
                            'lambda_d': self.lambda_params["d"],
                            'lambda_p': self.lambda_params["p"],
                            'delta_s': delta_s,
                            'delta_d': delta_d,
                            'delta_p': delta_p,
                            't5': self.t5, 'clip': self.clip, 'ae': self.ae,
                            'device': self.accelerator.device, 'weight_dtype': self.weight_dtype,
                            'use_gen_prob': current_use_gen_prob,
                            'base_model_name': self.config.model.base_model_name,
                        }

                        loss, img_loss, txt_s_loss, txt_d_loss, txt_p_loss, c_s, c_d, c_p = train_one_step(**dict_input)
                        avg_loss = self.accelerator.gather(loss.repeat(self.batch_size)).mean()
                        avg_img_loss = self.accelerator.gather(img_loss.repeat(self.batch_size)).mean()
                        avg_txt_s_loss = self.accelerator.gather(txt_s_loss.repeat(self.batch_size)).mean()
                        avg_txt_d_loss = self.accelerator.gather(txt_d_loss.repeat(self.batch_size)).mean()
                        avg_txt_p_loss = self.accelerator.gather(txt_p_loss.repeat(self.batch_size)).mean()
                        train_loss += avg_loss.item() / self.config.training.gradient_accumulation_steps
                        train_img_loss += avg_img_loss.item() / self.config.training.gradient_accumulation_steps
                        train_txt_s_loss += avg_txt_s_loss.item() / self.config.training.gradient_accumulation_steps
                        train_txt_d_loss += avg_txt_d_loss.item() / self.config.training.gradient_accumulation_steps
                        train_txt_p_loss += avg_txt_p_loss.item() / self.config.training.gradient_accumulation_steps

                        self.accelerator.backward(loss)
                        if self.accelerator.sync_gradients:
                            self.accelerator.clip_grad_norm_(self.model.parameters(), self.config.training.max_grad_norm)
                        self.optimizer.step()
                        if self.accelerator.sync_gradients:
                            self.lr_scheduler.step()
                            self._apply_lr_floor()
                        self.optimizer.zero_grad()

                        if self.accelerator.sync_gradients and len(self.lambda_params) > 0:
                            lambda_loss = -(self.lambda_params["s"] * c_s.detach()
                                            + self.lambda_params["d"] * c_d.detach()
                                            + self.lambda_params["p"] * c_p.detach())
                            self.lambda_optimizer.zero_grad()
                            self.accelerator.backward(lambda_loss)
                            self.lambda_optimizer.step()
                            with torch.no_grad():
                                for key in ("s", "d", "p"):
                                    self.lambda_params[key].clamp_(min=0.0)

                    if self.accelerator.sync_gradients:
                        current_lr = self.lr_scheduler.get_last_lr()[0]
                        logs = {"loss": train_loss, "lr": current_lr, "use_gen_prob": current_use_gen_prob}
                        for key in ("s", "d", "p"):
                            param = getattr(self, f"lambda_param_{key}", None)
                            if param is None:
                                continue
                            logs[f"lambda_{key}"] = param.item()
                        logs["c_s"] = c_s.detach().item()
                        logs["c_d"] = c_d.detach().item()
                        logs["c_p"] = c_p.detach().item()

                        log_dict = {
                            "loss": train_loss,
                            "img_loss": train_img_loss,
                            "txt_s_loss": train_txt_s_loss,
                            "txt_d_loss": train_txt_d_loss,
                            "txt_p_loss": train_txt_p_loss,
                            "learning_rate": current_lr,
                            "use_gen_prob": current_use_gen_prob,
                            "c_s": c_s.detach().item(),
                            "c_d": c_d.detach().item(),
                            "c_p": c_p.detach().item(),
                        }

                        for key in ("s", "d", "p"):
                            param = getattr(self, f"lambda_param_{key}", None)
                            if param is None:
                                continue
                            log_dict[f"lambda_{key}"] = param.item()

                        self.accelerator.log(log_dict, step=self.global_step)

                        train_loss = 0.0
                        train_img_loss = 0.0
                        train_txt_s_loss = 0.0
                        train_txt_d_loss = 0.0
                        train_txt_p_loss = 0.0

                        progress_bar.set_postfix(**logs)

                        if self.accelerator.is_main_process:
                            if self.global_step % self.config.training.val_steps == 0:
                                self._run_validation(batch)

                            if self.global_step % self.config.training.save_steps == 0:
                                self._save_training_state()

                        self.global_step += 1
                        progress_bar.update(1)

                self._finish_stage(stage_idx, stage_iterator)

        except Exception as e:
            self.logger.error(f"Training error at step {self.global_step}: {e}", exc_info=True)
            import sys
            import traceback
            print(f"ERROR in training loop at step {self.global_step}: {type(e).__name__}: {str(e)}", file=sys.stderr)
            print("Full traceback:", file=sys.stderr)
            traceback.print_exc(file=sys.stderr)
            sys.stderr.flush()
            raise
        finally:
            progress_bar.close()
            self.logger.info(f"Training completed! Reached max_train_steps: {self.total_iterations}")
            self.accelerator.wait_for_everyone()
            self.accelerator.end_training()

    def _start_stage(self, stage_idx: int, stage_end: int):
        total_stages = len(self.config.training.iterations)
        stage_batch = self.config.training.batch_sizes[stage_idx]
        stage_patch = self.config.training.patch_sizes[stage_idx]
        self.logger.info(
            f"Starting stage {stage_idx + 1}/{total_stages}: target_step={stage_end}, "
            f"batch_size={stage_batch}, patch_size={stage_patch}"
        )

        self.batch_size = stage_batch
        self.patch_size = stage_patch

        self.cleanup_shared_memory()
        iterator = self.load_data()
        self.accelerator.wait_for_everyone()
        return iterator

    def _finish_stage(self, stage_idx: int, iterator):
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception as exc:  # noqa: B902
                self.logger.warning(f"Failed to shutdown dataloader workers gracefully: {exc}")
        self.cleanup_shared_memory()
        self.accelerator.wait_for_everyone()
        self.logger.info(
            f"Completed stage {stage_idx + 1} at global step {self.global_step}"
        )

    def _compute_use_gen_prob(self) -> float:
        prob = 0.0
        if getattr(self, "use_gen", None):
            for step, value in sorted(self.use_gen.items()):
                if self.global_step >= step:
                    prob = value
                else:
                    break
        return prob

    def _apply_lr_floor(self):
        lr_floor = getattr(self.config.training, "min_learning_rate", 5e-5)
        for group in self.optimizer.param_groups:
            if group['lr'] < lr_floor:
                group['lr'] = lr_floor

    def _run_validation(self, latest_batch):
        self.logger.info(f"Running validation at step {self.global_step}")
        self.model.eval()
        val_bs = self.config.data.val_batch_size

        def _slice_item(item):
            if isinstance(item, torch.Tensor):
                return item[:val_bs]
            if hasattr(item, '__getitem__'):
                return item[:val_bs]
            return item

        val_batch = tuple(_slice_item(comp) for comp in latest_batch)

        val_dict_input = {
            'model': self.model,
            'batch': val_batch,
            'args': self.config,
            't5': self.t5,
            'clip': self.clip,
            'ae': self.ae,
            'device': self.accelerator.device,
            'weight_dtype': self.weight_dtype,
        }

        try:
            val_results = val_one_step(**val_dict_input)
            visualize_results(
                val_results,
                save_dir=os.path.join(self.config.data.output_dir, 'visualization'),
                step=self.global_step
            )
        except Exception as exc:  # noqa: B902
            self.logger.warning(f"Validation visualization failed (non-critical): {exc}")
            self.logger.info("Validation images may still be saved locally. Continuing training...")
        finally:
            self.model.train()

    def _save_training_state(self):
        save_checkpoint(
            model=self.model,
            optimizer=self.optimizer,
            lr_scheduler=self.lr_scheduler,
            global_step=self.global_step,
            output_dir=self.config.data.output_dir,
            checkpoints_total_limit=self.config.training.checkpoints_total_limit,
            logger=self.logger,
            lambda_param=self.lambda_params if len(self.lambda_params) > 0 else None,
            lambda_optimizer=self.lambda_optimizer if self.lambda_optimizer is not None else None,
        )