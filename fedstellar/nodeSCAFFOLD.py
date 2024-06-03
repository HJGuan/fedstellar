# 
# This file defines nodes for the implementation of FedEP algorithms.
#
import torch
import numpy as np
import math
import time

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.lightninglearnerSCAFFOLD import LightningLearnerSCAFFOLD
from fedstellar.node import Node
import threading
import logging
from fedstellar.messages import LearningNodeMessages
from fedstellar.role import Role

from fedstellar.proto import node_pb2, node_pb2_grpc
import grpc


class NodeSCAFFOLD(Node):

    epsilon_prime = 0.1

    def __init__(self,
            idx,
            experiment_name,
            model,
            data,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearnerSCAFFOLD,
            encrypt=False,
            model_poisoning=False,
            poisoned_ratio=0,
            noise_type='gaussian',
    ):
        Node.__init__(self, idx, experiment_name, model, data, host, port, config, learner, encrypt, model_poisoning, poisoned_ratio, noise_type)

        # FedEP specific parameters
        logging.info(f"[NodeSCAFFOLD] Initializing SCAFFOLD node")
        print(f"[NodeSCAFFOLD] Initializing SCAFFOLD node")

        self._control_variabtes = None
        self.learner = LightningLearnerSCAFFOLD(model, data, config, self.logger)

    ###########################################################################################
    #    SCAFFOLD                                                                             #
    ###########################################################################################

    def __train_step(self):
        # Set train set
        if self.round is not None:
            # self.__train_set = self.__vote_train_set()
            self.__train_set = self.get_neighbors(only_direct=False)
            self.__train_set = self.__validate_train_set(self.__train_set)
            if self.addr not in self.__train_set:
                self.__train_set.append(self.addr)
            logging.info(
                f"{self.addr} Train set: {self.__train_set}"
            )
            # Logging neighbors (indicate the direct neighbors and undirected neighbors)
            logging.info(
                f"{self.addr} Direct neighbors: {self.get_neighbors(only_direct=True)} | Undirected neighbors: {self.get_neighbors(only_undirected=True)}"
            )

        # Determine if node is in the train set
        if self.config.participant["device_args"]["role"] == Role.AGGREGATOR:
            logging.info("[NODE.__train_step] Role.AGGREGATOR process...")
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)

            # Evaluate
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()



            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model_scaffold(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                    self.learner.get_client_controls(),
                    source=self.addr,
                    round=self.round
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED, models_added
                    )
                )
                
                self.__gossip_model_aggregation()
                
        elif self.config.participant["device_args"]["role"] == Role.SERVER:
            logging.info("[NODE.__train_step] Role.SERVER process...")
            logging.info(f"({self.addr}) Model hash start: {self.learner.get_hash_model()}")
            # No train, evaluate, aggregate the models and send model to the trainer node
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)
                
            # Evaluate
            if self.round is not None:
                self.__evaluate()
                
            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model_scaffold(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                    self.learner.get_client_controls(),
                    source=self.addr,
                    round=self.round
                )
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                        LearningNodeMessages.MODELS_AGGREGATED, models_added
                    )
                )
                self.__gossip_model_aggregation()
                
        elif self.config.participant["device_args"]["role"] == Role.TRAINER:
            logging.info("[NODE.__train_step] Role.TRAINER process...")
            logging.info(f"({self.addr}) Model hash start: {self.learner.get_hash_model()}")
            if self.round is not None:
                # Set Models To Aggregate
                self.aggregator.set_nodes_to_aggregate(self.__train_set)
                logging.info(f"({self.addr}) Waiting aggregation | Assign __waiting_aggregated_model = True")
                self.aggregator.set_waiting_aggregated_model(self.__train_set)

            # Evaluate
            if self.round is not None:
                self.__evaluate()

            # Train
            if self.round is not None:
                self.__train()
                logging.info(f"({self.addr}) Model hash after local training: {self.learner.get_hash_model()}")

            # Aggregate Model
            if self.round is not None:
                models_added = self.aggregator.add_model_scaffold(
                    self.learner.get_parameters(),
                    [self.addr],
                    self.learner.get_num_samples()[0],
                    self.learner.get_client_controls(),
                    source=self.addr,
                    round=self.round,
                    local=True
                )
                
                # send model added msg ---->> redundant (a node always owns its model)
                self._neighbors.broadcast_msg(
                    self._neighbors.build_msg(
                       LearningNodeMessages.MODELS_AGGREGATED, models_added
                   )
                )
                                
                logging.info(f"({self.addr}) Gossiping (difusion) my current model parameters.")
                self.__gossip_model_difusion()


        elif self.config.participant["device_args"]["role"] == Role.IDLE:
            # If the received model has the __train_set as contributors, then the node overwrites its model with the received one
            logging.info("[NODE.__train_step] Role.IDLE process...")
            # Set Models To Aggregate
            self.aggregator.set_nodes_to_aggregate(self.__train_set)
            logging.info(f"({self.addr}) Waiting aggregation.")
            self.aggregator.set_waiting_aggregated_model(self.__train_set)

        else:
            logging.warning("[NODE.__train_step] Role not implemented yet")

        # Gossip aggregated model
        if self.round is not None:
            logging.info(f"({self.addr}) Waiting aggregation and gossiping model (difusion).")
            self.__wait_aggregated_model()
            self.__gossip_model_difusion()

        # Finish round
        if self.round is not None:
            self.__on_round_finished()










    ############################
    #  GRPC - Remote Services  #
    ############################
    
    def add_model_scaffold(self, request, context):
        """
        """
        try:
            scaffoldWeights = self.learner.decode_parameters(request.scaffoldWeights)
            print(f"[SCAFFOLD] : {scaffoldWeights}")
            received_weight = next(iter(scaffoldWeights.values()))
            self.aggregator.add_model_scaffold(
                models=scaffoldWeights,
                addr=[request.source],
                weight=scaffoldWeights,
                all_neighbors = self._neighbors.get_all()
            )
            return node_pb2.ResponseMessage(error="")
        except Exception as e:
            return node_pb2.ResponseMessage(error=str(e))








        
       
    

       
 