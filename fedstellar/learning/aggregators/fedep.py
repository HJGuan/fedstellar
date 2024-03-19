# 
# FEDEP: Federated Entropy Pooling
# 


import logging
import threading

import torch
import numpy as np

from fedstellar.learning.aggregators.aggregator import Aggregator


class FedEP(Aggregator):
    """

    """

    def __init__(self, node_name="unknown", config=None):
        super().__init__(node_name, config)
        self.config = config
        self.role = self.config.participant["device_args"]["role"]
        self._waiting_distribution = False
        self._distributions = {}
        self._thetas_with_samples_num = {}
        self._prediction_precision = 1e-3
        self._prediction_epsilon = 1e-2
        self._gaussian_epsilon = 1e-1
        self._theta_global = None
        self._prob_global = None
        self._clients_probs = None
        self._KL_divergence = None
        self._labels = None
        self._labels_unique = None
        self.alpha_k = None
        self._lock_to_start_pooling = threading.Lock()

    def set_waiting_distribution(self, TrueOrFalse):
        print(f"(_waiting_distribution set to {TrueOrFalse})")
        self._waiting_distribution = TrueOrFalse

    def _gaussian(self, Y, mu, sigma_squared):
        '''
        probability density function of Gaussian distribution
        '''
        return np.exp(-np.square(Y-mu+self._gaussian_epsilon)/(2*(sigma_squared+self._gaussian_epsilon)))/(np.sqrt(2 * np.pi * (sigma_squared + self._gaussian_epsilon)))

    def aggregate(self, models):
        """
        Ponderated average of the models.

        Args:
            models: Dictionary with the models (node: model,num_samples).
            model : {layer: tensor, ...}
        """
        # Check if there are models to aggregate
        if len(models) == 0:
            logging.error(
                "[FedEP] Trying to aggregate models when there is no models"
            )
            return None
        
        print(f"[FedEP.aggregate] Aggregating models: num={len(models)}")

        # Create a shape of the weights use by all nodes
        accum = {layer: torch.zeros_like(param) for layer, param in list(models.values())[-1][0].items()}

        # Add weighted models
        
        for address, model in models.items():
            print(f"accumulating address: {address}")
            for layer in accum:
                print(f"model[0][{layer}]: {model[0][layer]}")
                accum[layer] += model[0][layer] * self.alpha_k[address]
            
        return accum

    
    def _predict_likelihood(self, theta, precision=4):
        '''
        given theta and labels, calculate the likelihood of the labels
        '''
        prob = np.zeros(len(self._labels_unique))
        for i in range(len(prob)): 
            prob[i] = np.round(np.sum(theta[:,0] * self._gaussian(self._labels_unique[i], theta[:,1], theta[:,2])), precision)
        return prob
    
    def pooling(self,labels):
        '''
        Examples:

        self._theta_global:
        {'10.10.10.10': [[0.2*(1/3), 9.0, 0.1], [0.8*(1/3), 1.18181818, 1.33]],
         '9.9.9.9': [[0.3*(2/3), 4.0, 0.1], [0.7*(2/3), 5.118, 1.33]]}

        self._prob_global:
        [0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.7] (len(self._prob_global)=10 && sum(self._prob_global)=1)


        self._clients_probs:
        {'10.10.10.10': [0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0, 0.7],
         '9.9.9.9':[ 0, 0, 0, 0, 0.1, 0.1, 0.1, 0, 0, 0.7]}

        self._KL_divergence:
        {'10.10.10.10': 3, '9.9.9.9': 2}

        self.alpha_k:
        {'10.10.10.10': 3/(2+3), '9.9.9.9': 2/(2+3)}

        '''
        self._lock_to_start_pooling.acquire()

        print(f"[nodeFedEP] Pooling started")
        self._labels = labels
        self._labels_unique = np.unique(labels)
        # Total Samples
        total_samples = sum([weight for _, weight in self._thetas_with_samples_num.values()])
        print(f"[FedEP]total_samples: {total_samples}")

        q_k = {[k][0]: v[1]/total_samples for k, v in self._thetas_with_samples_num.items()}
        
        self._theta_global = {
            address: np.array([
                [round(param[0] * (weight / total_samples), 5), param[1], param[2]]
                for param in theta
            ])
            for address, (theta, weight) in self._thetas_with_samples_num.items()
        }
        print(f"[FedEP]theta_global: {self._theta_global}")
        logging.info(f"theta_global: {self._theta_global}")

        # Calculate global probability of labels
        '''
        self._prob_global = [] with a length equals number of labels space and summing up to 1
        '''
        prob_global = []
        for label in self._labels_unique:
            prob_label = 0
            for theta in self._theta_global.values():
                for g in theta:
                    prob_label +=  g[0] * self._gaussian(label, g[1], g[2]) 
            prob_global.append(prob_label)
        self._prob_global = prob_global

        # Calculate client probabilities
        self._clients_probs = { 
            address : self._predict_likelihood(theta)
            for address,theta in self._theta_global.items()
        }
        print(f"[FedEP]clients_probs: {self._clients_probs}")
        
        # Calculate KL divergences
        self._KL_divergence = {
            address: np.sum([self._prob_global[i] * np.log2((self._prob_global[i] + self._prediction_epsilon) / (probs[i] + self._prediction_epsilon)) for i in range(len(self._prob_global))])
            for address,probs in self._clients_probs.items()
        }
        logging.info(f"[FedEP]KL_divergence: {self._KL_divergence}")

        # Calculate alpha_k
        kl_sum = np.sum([kl_div for kl_div in self._KL_divergence.values()])
        self.alpha_k = {
            address: kl_div / kl_sum
            for address, kl_div in self._KL_divergence.items()
        }
        print(f"[FedEP]alpha_k: {self.alpha_k}")
        logging.info(f"[FedEP]alpha_k: {self.alpha_k}")

        print(f"[nodeFedEP] Pooling ended")

    
    def add_distribution(self, theta, addr, weight, all_neighbors):
        """
        Add a distribution. The first model to be added starts the `run` method (timeout).

        Args:
            theta: local distribution of this client, represented by theta (a 3xM array).
            contributors: Nodes that collaborated to get the model.
            weight: Number of samples used to get the model.

        self._thetas_with_samples_num:
        {'10.10.10.10': ([[0.2, 9.0, 0.1], [0.8, 1.18181818, 1.33]], 100),
         '9.9.9.9': ([[0.3, 4.0, 0.1], [0.7, 5.118, 1.33]], 200)}
        """
        print(f"({self.node_name}) adding distribution...")

        self._thetas_with_samples_num["".join(addr)] = (theta, weight)
        print(f"[FedEP] Current Distributions: {self._thetas_with_samples_num}")

         # all distributions recieved
        if len(self._thetas_with_samples_num) > len(all_neighbors):
            print(f"({self.node_name}) all distributions received")
            # self.set_waiting_distribution(False)
            self._lock_to_start_pooling.release()
        else:
            print(f"({self.node_name}) waiting for more distributions ({len(self._thetas_with_samples_num)}/{len(all_neighbors)+1})")

        return self._thetas_with_samples_num
    

 
        

    





        


        
    