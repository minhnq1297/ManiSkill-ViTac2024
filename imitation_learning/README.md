This branch is the attempt to apply [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) to contact-rich robot manipulation tasks using visuotactile sensors

## Before Starting
Please follow the instruction [here](../README.md) to install the environment properly, then activate the `conda` environment with `conda activate mani_vitac`

We assume that, before you execute any command, you are at the root of the repo

If some packages are not found during execution of a script, simply install them with `pip install package_name`

## Installing Diffusion Policy
Installing [Diffusion Policy](https://github.com/real-stanford/diffusion_policy)
```
cd ..
git clone https://github.com/real-stanford/diffusion_policy.git
cd diffusion_policy
pip install -e .
```

## Collecting Expert Data
Replace `YOUR_NUMBER` with the number of offsets(i.e. initial settings) for each task that you want to generate expert data. Its default value is `1500`. After finishing, the data is stored under `./data`

For `peg_insertion` task
```
python imitation_learning/data_generation/open_lock_demo_generation.py --num_of_offset=YOUR_NUMBER_HERE
```
For `open_lock` task
```
python imitation_learning/data_generation/open_lock_demo_generation.py --num_of_offset=YOUR_NUMBER_HERE
```

## Training Diffusion Policy
There are some important notes:
- `pred_horizon` variable here is not the same as prediction horizon defined in the original [paper](https://diffusion-policy.cs.columbia.edu/#paper). `pred_horizon` here means the number of actions that the policy will predict from the current time step. In another word, `pred_horizon` + `obs_horizon` is equal to prediction horizon in the original paper
- The backend networks from [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) package do not accept every value of `pred_horizon` and `obs_horizon`. For `peg_insertion` task, we recommend `pred_horizon = 2, 6`, `obs_horizon=2`. For `open_lock` task, `pred_horizon = 4, 8`, `obs_horizon=4`

To see all options and parameters which could be used with the command, one can do
```
python imitation_learning/diffusion_policy/train_diffusion_policy.py --h
```
Some of them are:
- `--task_name`: the value must be either "peg_insertion" or "open_lock"
- `--train_data_path`: the path to your collected expert data
- `--obs_horizon`: observation horizon as described in the original paper
- `--pred_horizon`: prediction horizon as explained above
- `--use_pretrained_encoder`: `1` to use pretrained marker encoder, `0` to end-to-end with your data
- `--use_ee_pose`: `1` to include end-effector (EEF) pose, `0` to exclude it
- `--use_wandb`: `1` to log your training process to [wandb](https://wandb.ai), `0` to disable it

Start the training by providing all necessary arguments to the script, for example
```
python imitation_learning/diffusion_policy/train_diffusion_policy.py --task_name="open_lock" --train_data_path="./data/open_lock_demo-20240726-012658.pkl.gzip" --obs_horizon=4 --pred_horizon=4 --learning_rate=1e-4 --n_epoch=2500 --batch_size=256 --use_ee_pose=1 --use_pretrained_encoder=1 --use_wandb=1 
```
When the training is done, the model  is stored under `./trained_model`.

## Testing Diffusion Policy
There is a test script for each task. The scripts take some arguments as follow
- `--trained_model_path`: the path to your trained model
- `--obs_horizon`: same meaning as above, and its value must be the same as you had for your training
- `--pred_horizon`: Prediction horizon as explained above. You can vary this during test time, but its value must be bigger than the value of `pred_horizon` you had for your training
- `--action_horizon`: number of action that agent would execute without replanning. This must be less than `pred_horizon`
- `--use_pretrained_encoder`: the value must be the same as you had for your training
- `--use_ee_pose`: the value must be the same as you had for your training

In general, Start the testing by providing all necessary arguments to the script. Two examples could be
- Testing `peg_insertion` task
    ```
    python imitation_learning/diffusion_policy/test_peg_insertion.py --trained_model_path="./trained_model/checkpoint_peg_insertion_model_20240731-002836.pth.tar" --obs_horizon=2 --action_horizon=1 --pred_horizon=6 --use_pretrained_encoder=1 --use_ee_pose=0
    ```

- Testing `open_lock` task
    ```
    python imitation_learning/diffusion_policy/test_open_lock.py --trained_model_path="./trained_model/checkpoint_open_lock_model_20240731-014612.pth.tar" --obs_horizon=4 --action_horizon=1 --pred_horizon=4 --use_pretrained_encoder=1 --use_ee_pose=1
    ```
After testing, the log is saved to `./eval_log`