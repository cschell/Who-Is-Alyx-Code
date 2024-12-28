import pathlib

from src.utils import utils

log = utils.get_logger(__name__)


def upload_config_and_checkpoints_to_wandb(datamodule):
    import wandb
    checkpoint_dir = pathlib.Path("checkpoints/")
    if checkpoint_dir.exists():
        wandb.save(f"{checkpoint_dir}/*.ckpt")
    else:
        log.warning(f"checkpoint directory {checkpoint_dir} does not exists, so no checkpoints will be uploaded to wandb.io")
    if datamodule.data_stats_path.exists():
        wandb.save(str(datamodule.data_stats_path))
    else:
        log.warning(f"data stats file with training stats {datamodule.data_stats_path} does not exists, so it cannot be uploaded to wandb.io")
    wandb.save(".hydra/*.yaml")


def is_wandb_logger_enabled(loggers):
    # for lg in loggers:
    #     if isinstance(lg, pl.loggers.wandb.WandbLogger):
    #         return True
    # return False
    return True
