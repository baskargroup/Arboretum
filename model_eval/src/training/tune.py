import ast
import collections, yaml
from pathlib import Path
import argparse
import os

import ray
from ray import air, tune
from ray.air.integrations.wandb import WandbLoggerCallback

import wandb

from main import run_main, yaml_setup

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config-yaml",
        type=str,
        default=None,
        help="Path to base config yaml",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=16,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        default=1,
        help="Number of epochs to train for",
    )
    parser.add_argument(
        "--distributed",
        default=False,
        action="store_true",
        help="Use distributed arguments for tuning."
    )
    parser.add_argument(
        "--debug",
        default=False,
        action="store_true",
        help="Detailed logs for debugging."
    )
    parser.add_argument(
        "--redis-password",
        type=str,
        default=None,
        help="redis password for ray",
    )
    args = parser.parse_args()
    return args

def run_tune():
    num_cpus = int(os.getenv("SLURM_CPUS_PER_TASK"))
    #num_gpus = int(os.getenv("SLURM_GPUS_PER_TASK"))
    port_address = os.getenv("ip_head")
    yaml_setup()
    args = parse_args()
    if args.debug:
        os.environ["RAY_LOG_TO_STDERR"] = "1"
    with open( args.config_yaml , 'r' ) as stream:
        tr_config = yaml.safe_load( stream )
    # kp = Path("/scratch/bf996/vlhub/debug/wandb.txt")
    # with open( kp, 'r' ) as file:
    #     wandb_key = ast.literal_eval( file.read( ) )
    tr_config.update({"lr" : tune.grid_search([2e-4,3e-4,4e-4]),
        "beta1" : tune.grid_search([0.8,0.9,0.99]),
        "beta2" : tune.grid_search([0.8,0.9,0.99]),
        "epochs": args.epochs,
        "local_loss": args.distributed,
        "gather_with_grad": args.distributed,
        "zeroshot_frequency": 1,
        "raytune": True,
        "report-to": "wandb"})
    # tr_config = {
    #     "lr" : tune.grid_search([2e-4,3e-4,4e-4,1e-4]),
    #     "beta1" :  tune.grid_search([0.5,0.8,0.9,0.99]),
    #     "beta2" : tune.grid_search([0.5,0.8,0.9,0.99]),
    #     "epochs":32,
	# 			# specify wandb project and apikey
    #     "wandb": {
    #         "project": "open-clip",
    #         "api_key": wandb_key,
    #     }
    # }
    #ray.init(address=port_address, _redis_password=args.redis_password, ignore_reinit_error=True)
    ray.init(address=port_address, ignore_reinit_error=True, num_cpus=num_cpus, num_gpus=args.num_gpus)
    tuner = tune.Tuner(
        # run_main,
        tune.with_resources(run_main, resources={"cpu": num_cpus, "gpu": args.num_gpus}),
        tune_config=tune.TuneConfig(
            metric="accuracy",
            mode="max",
        ),
        run_config=air.RunConfig(
            callbacks=[
                WandbLoggerCallback(project="open-clip")
            ]
        ),
        param_space=tr_config,
    )
    tuner.fit()
    ray.shutdown()
    print("Hyperparameter sweep complete")


if __name__ == "__main__":
    run_tune()