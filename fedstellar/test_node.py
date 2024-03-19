import os
import sys
import time
import random
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from fedstellar.node import Node, MaliciousNode
from fedstellar.nodeFedEP import NodeFedEP
from fedstellar.learning.pytorch.mnist.mnist import MNISTDataset
from fedstellar.config.config import Config
from fedstellar.learning.pytorch.mnist.models.mlp import MNISTModelMLP


def main():
    node = Node(
        idx=0,
        experiment_name="experiment_name",
        model=MNISTModelMLP(),
        data= MNISTDataset(num_classes=10, sub_id=0, number_sub=2, iid=True, partition="percent", seed=42, config=None),
        host="192.168.50.9",
        port=45000,
        config=None,
        encrypt=False,
        model_poisoning=False,
        poisoned_ratio=0,
        noise_type="salt"
    )

    node2 = NodeFedEP(
        idx=0,
        experiment_name="experiment_name",
        model=MNISTModelMLP(),
        data= MNISTDataset(num_classes=10, sub_id=1, number_sub=2, iid=True, partition="percent", seed=42, config=None),
        host="192.168.50.9",
        port=45000,
        config=None,
        encrypt=False,
        model_poisoning=False,
        poisoned_ratio=0,
        noise_type="salt"
    )

    print(node2.data.targets)


if __name__ == "__main__":
    main()

