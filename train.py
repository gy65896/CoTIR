import argparse
import traceback
import sys
from omegaconf import OmegaConf
from trainer import Trainer

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Simple example of a training script.")
        parser.add_argument(
            "--config",
            type=str,
            default='./configs/train_cotir-9b.yaml',
            help="path to config",
        )
        args = parser.parse_args()
        config = OmegaConf.load(args.config)

        trainer = Trainer(config)
        trainer.start_training()
    except Exception as e:
        # Print full traceback to stderr so it's visible in distributed training
        print(f"ERROR: {type(e).__name__}: {str(e)}", file=sys.stderr)
        print("Full traceback:", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.stderr.flush()
        raise