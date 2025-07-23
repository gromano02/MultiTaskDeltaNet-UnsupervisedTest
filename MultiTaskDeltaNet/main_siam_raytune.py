import numpy as np # linear algebra
import os

from models.trainer_siam_raytune import *

import ray
from ray import tune, train
from ray.train import Checkpoint
from ray.tune.schedulers import ASHAScheduler
from functools import partial


os.environ["RAY_TMPDIR"] = "/tmp/my_tmp_dir"
ray.init(_temp_dir="/tmp/my_tmp_dir")


def main(num_samples=10, max_num_epochs=200, gpus_per_trial=1):

    config = {
        "lr": tune.loguniform(1e-5, 1e-2),
        "batch_size": tune.choice([ 4, 8, 16]),
    }

    scheduler = ASHAScheduler(
        #metric="loss",
        #mode="min",
        metric="f1",
        mode='max',
        max_t=max_num_epochs,
        grace_period=10,       #15,20     ####train each trial at least for n epochs
        reduction_factor=2,)   ####if =2, only 50% of all trials are kept each time they are reduced

    tuner = tune.Tuner(
        tune.with_resources(
            #tune.with_parameters(train_cifar),
            partial(CDTrainer),  
            resources={"cpu": 2, "gpu": gpus_per_trial}
        ),    
        tune_config=tune.TuneConfig(
            #metric="loss",
            #mode="min",
            scheduler=scheduler,
            num_samples=num_samples,
        ),
        param_space=config,
        run_config=train.RunConfig(
            checkpoint_config=train.CheckpointConfig(
            checkpoint_at_end=False,
            #checkpoint_frequency=2,
            checkpoint_score_attribute="f1",
            num_to_keep=10,
            ),    
        ),   
    )

    result = tuner.fit()

    #best_trial = result.get_best_result("loss", "min")
    best_trial = result.get_best_result("f1", "max")
    print(f"Best trial config: {best_trial.config}")
    print(f"Best trial final validation loss: {best_trial.metrics['loss']}")
    print(f"Best trial final validation f1_1: {best_trial.metrics['f1_1']}")
    print(f"Best trial final validation f1_2: {best_trial.metrics['f1_2']}")
    print(f"Best trial final validation mean f1: {best_trial.metrics['f1']}")

    """
    dfs = {result.path: result.metrics_dataframe for result in results}
    fig = plt.figure(figsize = (25, 20))
    ax = None  # This plots everything on the same plot
    for d in dfs.values():
        #ax = d.f1.plot(ax=ax, legend=True)
        ax = d.f1.plot(ax=ax,label=d.trial_id[0],legend=False)

    ax.legend(fontsize = 15)
    ax.set_xlabel('epochs')
    ax.set_ylabel('f1 score')
    fig.savefig('full_figure.png')    
    """

if __name__ == "__main__":
    # You can change the number of GPUs per trial here:
    main(num_samples=100, max_num_epochs=500, gpus_per_trial=1) 

