# 
# FEDEP: Federated Entropy Pooling
# 


import logging
import threading

import torch
import numpy as np

from fedstellar.learning.aggregators.aggregator import Aggregator


class SCAFFOLD(Aggregator):
    """

    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]


    def aggregate(self, models):
        """
        Weighted average of the models. Same as FedAvg

        Args:
            models: Dictionary with the models (node: model, num_samples).
        """
        if len(models) == 0:
            logging.error("[FedAvg] Trying to aggregate models when there are no models")
            return None

        models = list(models.values())

        # Total Samples
        total_samples = sum(w for _, w in models)

        # Create a Zero Model
        accum = {layer: torch.zeros_like(param) for layer, param in models[-1][0].items()}

        # Add weighted models
        logging.info(f"[FedAvg.aggregate] Aggregating models: num={len(models)}")
        for model, weight in models:
            for layer in accum:
                accum[layer] += model[layer] * weight

        # Normalize Accum
        for layer in accum:
            accum[layer] /= total_samples
            
        # self.print_model_size(accum)

        return accum
    
    def add_model_scaffold(self, model, contributors, weight, ci, source=None, round=None, local=False):
        """
        Add a model. The first model to be added starts the `run` method (timeout).

        Args:
            model: Model to add.
            contributors: Nodes that collaborated to get the model.
            weight: Number of samples used to get the model.
            source: Node that sent the model.
            round: Round of the aggregation.
        """

        nodes = list(contributors)
        logging.info(
            f"({self.node_name}) add_model_scaffold (aggregator) | source={source} | __models={self.__models.keys()} | contributors={nodes} | train_set={self.__train_set} | get_aggregated_models={self.get_aggregated_models()}")

        # Verify that contributors are not empty
        if contributors == []:
            logging.info(
                f"({self.node_name}) Received a model without a list of contributors."
            )
            logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __agg_lock.")
            self.__agg_lock.release()
            return None

        # Check again if the round is the same as the current one, if not, ignore the model (it is from a previous round)
        if round != self.__round:
            logging.info(
                f"({self.node_name}) add_model_scaffold (aggregator) | Received a model from a previous round."
            )
            logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __agg_lock.")
            if self.__agg_lock.locked():
                self.__agg_lock.release()
            return None

        # Diffusion / Aggregation
        if self.__waiting_aggregated_model and not local:
            logging.info(
                    f"({self.node_name}) add_model_scaffold (aggregator) | __waiting_aggregated_model (True)")
            if set(contributors) == set(self.__train_set):
                logging.info(
                    f"({self.node_name}) add_model_scaffold (aggregator) | __waiting_aggregated_model (True) | Ignoring add_model_scaffold functionality...")
                logging.info(
                    f"({self.node_name}) add_model_scaffold (aggregator) | __waiting_aggregated_model (True) | Received an aggregated model because all contributors are in the train set (me too). Overwriting __models with the aggregated model.")
                self.__models = {}
                self.__models = {" ".join(nodes): (model, 1)}
                self.__waiting_aggregated_model = False
                logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __finish_aggregation_lock.")
                self.__finish_aggregation_lock.release()
                return contributors
            else:
                logging.info(
                    f"({self.node_name}) add_model_scaffold (aggregator) | __waiting_aggregated_model (True) | Ignoring add_model_scaffold functionality...")

        else:
            logging.info(
                f"({self.node_name}) add_model_scaffold (aggregator) | Acquiring __agg_lock."
            )
            self.__agg_lock.acquire()

            # Check if aggregation is needed
            if len(self.__train_set) > len(self.get_aggregated_models()):
                # Check if all nodes are in the train_set
                if all([n in self.__train_set for n in nodes]):
                    logging.info(
                        f'({self.node_name}) add_model_scaffold (aggregator) | All contributors are in the train set. Adding model.')
                    # Check if the model is a full/partial aggregation
                    if len(nodes) == len(self.__train_set):
                        logging.info(
                            f'({self.node_name}) add_model_scaffold (aggregator) | The number of contributors is equal to the number of nodes in the train set. --> Full aggregation.')
                        self.__models = {" ".join(nodes): (model, weight, ci)}
                        logging.info(
                            f"({self.node_name}) add_model_scaffold (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )
                        # Finish agg
                        logging.info(
                            f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __finish_aggregation_lock.")
                        self.__finish_aggregation_lock.release()
                        # Unlock and Return
                        logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __agg_lock.")
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif all([n not in self.get_aggregated_models() for n in nodes]):
                        logging.info(
                            f'({self.node_name}) add_model_scaffold (aggregator) | All contributors are not in the aggregated models. --> Partial aggregation.')
                        # Aggregate model
                        self.__models[" ".join(nodes)] = (model, weight, ci)
                        logging.info(
                            f"({self.node_name}) add_model_scaffold (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )

                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            logging.info(
                                f"({self.node_name}) add_model_scaffold (aggregator) | All models were added. Finishing aggregation."
                            )
                            logging.info(
                                f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __finish_aggregation_lock.")
                            self.__finish_aggregation_lock.release()

                        # Unlock and Return
                        logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __agg_lock.")
                        self.__agg_lock.release()
                        return self.get_aggregated_models()

                    elif any([n in self.get_aggregated_models() for n in nodes]):
                        logging.info(
                            f'({self.node_name}) BETA add_model_scaffold (aggregator) | Some contributors are in the aggregated models.')

                        logging.info(
                            f'({self.node_name}) BETA add_model_scaffold (aggregator) | __models={self.__models.keys()}')

                        # Obtain the list of nodes that are not in the aggregated models
                        nodes_not_in_aggregated_models = [n for n in nodes if n not in self.get_aggregated_models()]
                        logging.info(
                            f'({self.node_name}) BETA add_model_scaffold (aggregator) | nodes_not_in_aggregated_models={nodes_not_in_aggregated_models}')

                        # For each node that is not in the aggregated models, aggregate the model with the aggregated model
                        for n in nodes_not_in_aggregated_models:
                            self.__models[n] = (model, weight,ci)

                        logging.info(
                            f'({self.node_name}) BETA add_model_scaffold (aggregator) | __models={self.__models.keys()}')

                        logging.info(
                            f"({self.node_name}) BETA add_model_scaffold (aggregator) | Model added ({str(len(self.get_aggregated_models()))}/{str(len(self.__train_set))}) from {str(nodes)}"
                        )
                        logging.info(f"({self.node_name}) BETA add_model_scaffold (aggregator) | self.aggregated_models={self.get_aggregated_models()}")
                        # Check if all models were added
                        if len(self.get_aggregated_models()) >= len(self.__train_set):
                            logging.info(
                                f"({self.node_name}) BETA add_model_scaffold (aggregator) | All models were added. Finishing aggregation."
                            )
                            logging.info(
                                f"({self.node_name}) BETA add_model_scaffold (aggregator) | Releasing __finish_aggregation_lock.")
                            self.__finish_aggregation_lock.release()

                        # Unlock and Return
                        logging.info(f"({self.node_name}) BETA add_model_scaffold (aggregator) | Releasing __agg_lock.")
                        self.__agg_lock.release()
                        return self.get_aggregated_models()
                    else:
                        logging.info(
                            f"({self.node_name}) add_model_scaffold (aggregator) | Can't add a model that has already been added {nodes}"
                        )
                else:
                    logging.info(
                        f"({self.node_name}) add_model_scaffold (aggregator) | Can't add a model from a node ({nodes}) that is not in the training test."
                    )
            else:
                logging.info(
                    f"({self.node_name}) add_model_scaffold (aggregator) | Received a model when is not needed."
                )
            logging.info(f"({self.node_name}) add_model_scaffold (aggregator) | Releasing __agg_lock.")
            self.__agg_lock.release()
            return None




    ############################
    #  GRPC - Remote Services  #
    ############################

    