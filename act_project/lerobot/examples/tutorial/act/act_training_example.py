"""This script demonstrates how to train ACT Policy on a real-world dataset."""

from pathlib import Path

import torch

from lerobot.configs import FeatureType
from lerobot.datasets import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.policies import make_pre_post_processors
from lerobot.policies.act import ACTConfig, ACTPolicy
from lerobot.utils.feature_utils import dataset_to_policy_features


def make_delta_timestamps(delta_indices: list[int] | None, fps: int) -> list[float]:
    if delta_indices is None:
        return [0]

    return [i / fps for i in delta_indices]


def main():
    output_directory = Path("outputs/robot_learning_tutorial/act")
    output_directory.mkdir(parents=True, exist_ok=True)

    # Select your device
    device = torch.device("cuda")  # or "cuda" or "cpu"

    dataset_id = "lerobot/svla_so101_pickplace"

    # This specifies the inputs the model will be expecting and the outputs it will produce
    #
    # dataset_metadata.features —— 数据集 info.json 的 features 字段，结构是 dict：
    #   key   → 字段名
    #   value → {"dtype": "video"|"image"|"float32",   # 存储类型
    #            "shape": (3,480,640) 或 (6,),        # 单帧形状
    #            "names": ["shoulder_pan.pos", ...]}  # 仅 state/action 有，每维含义
    #
    # 典型 SO101 数据集里的 key：
    #   observation.images.top / .wrist   视觉观测，dtype=video，shape=(3,H,W)
    #   observation.state                 6 维关节状态，shape=(6,)
    #   action                            6 维关节目标，shape=(6,)
    #   timestamp / frame_index           秒 / 帧号，标量
    #   episode_index / index             episode 编号 / 全局帧号，标量
    #   task_index                        语言任务编号，标量
    #
    # 下面两行：dataset_to_policy_features 把标量列（timestamp/index 等）丢掉，
    # 只留 STATE/VISUAL/ACTION；再往下 L51 把 action 划为输出，其余归输入。

    dataset_metadata = LeRobotDatasetMetadata(dataset_id)  # 获取数据集元数据
    features = dataset_to_policy_features(
        dataset_metadata.features
    )  # 将数据集元数据转换为 ACTPolicy 配置，类似于接口函数

    output_features = {
        key: ft for key, ft in features.items() if ft.type is FeatureType.ACTION
    }
    input_features = {
        key: ft for key, ft in features.items() if key not in output_features
    }

    cfg = ACTConfig(input_features=input_features, output_features=output_features)
    policy = ACTPolicy(cfg)  # policy里面初始化了一个actmodel
    preprocessor, postprocessor = make_pre_post_processors(#用于将数据集中的数据转换为actmodel输入的格式
        cfg, dataset_stats=dataset_metadata.stats
    )

    policy.train()
    policy.to(device)

    # To perform action chunking, ACT expects a given number of actions as targets
    delta_timestamps = {
        "action": make_delta_timestamps(
            cfg.action_delta_indices, dataset_metadata.fps
        ),  # 未来n帧的动作的时间间隔
    }

    # add image features if they are present
    delta_timestamps |= {
        k: make_delta_timestamps(
            cfg.observation_delta_indices, dataset_metadata.fps
        )  # 对cfg.image_features中的每个特征，计算未来n帧的时间间隔，
        for k in cfg.image_features
    }

    # Instantiate the dataset
    """
    当你把这个字典传给 LeRobotDataset 时，等于在数据集内部设定了一个“自动切片机”。

    如果不传这个参数：你每次 dataset[0]，拿到的是第 0 帧的当前图像、当前状态、当前动作（单步）。

    传了这个参数：你每次 dataset[0]，它会在后台自动根据你给的时间戳，把“过去几帧的图像”和“未来 100 步的动作块”全部给你切出来，打包成一个大字典返回。
    """
    dataset = LeRobotDataset(dataset_id, delta_timestamps=delta_timestamps)

    # Create the optimizer and dataloader for offline training
    optimizer = cfg.get_optimizer_preset().build(policy.parameters())
    batch_size = 32
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=device.type != "cpu",
        drop_last=True,
    )

    # Number of training steps and logging frequency
    training_steps = 1
    log_freq = 1

    # Run training loop
    step = 0
    done = False
    while not done:
        for batch in dataloader:
            batch = preprocessor(batch)
            loss, _ = policy.forward(batch)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            if step % log_freq == 0:
                print(f"step: {step} loss: {loss.item():.3f}")
            step += 1
            if step >= training_steps:
                done = True
                break

    # Save the policy checkpoint, alongside the pre/post processors
    policy.save_pretrained(output_directory)
    preprocessor.save_pretrained(output_directory)
    postprocessor.save_pretrained(output_directory)

    # Save all assets to the Hub
    policy.push_to_hub("<user>/robot_learning_tutorial_act")
    preprocessor.push_to_hub("<user>/robot_learning_tutorial_act")
    postprocessor.push_to_hub("<user>/robot_learning_tutorial_act")


if __name__ == "__main__":
    main()
