# 
# This file defines nodes for the implementation of FedEP algorithms.
#
import torch
import numpy as np
import math
import time

from fedstellar.config.config import Config
from fedstellar.learning.pytorch.lightninglearner import LightningLearner
from fedstellar.node import Node
import threading
import logging
from fedstellar.messages import LearningNodeMessages
from fedstellar.role import Role

from fedstellar.proto import node_pb2, node_pb2_grpc
import grpc


class NodeFedEP(Node):

    epsilon_prime = 0.1

    def __init__(self,
            idx,
            experiment_name,
            model,
            data,
            host="127.0.0.1",
            port=None,
            config=Config,
            learner=LightningLearner,
            encrypt=False,
            model_poisoning=False,
            poisoned_ratio=0,
            noise_type='gaussian',
    ):
        Node.__init__(self, idx, experiment_name, model, data, host, port, config, learner, encrypt, model_poisoning, poisoned_ratio, noise_type)

        # FedEP specific parameters
        logging.info(f"[NodeFedEP] Initializing FedEP node")
        print(f"[NodeFedEP] Initializing FedEP node")

        self._labels = np.array([label for _, label in self.learner.data.train_dataloader().dataset])

        self._distribution_fitting_max_components_fraction = 0.5
        self._EM_algorithm_max_iterations = 1500
        self._EM_algorithm_epsilon = 1e-6
        self._gaussian_epsilon = 1e-2

        self.theta = None

        # additional locks for FedEP
        self._distribution_communication_lock = threading.Lock()


    #########################################################################################################################################################
    #    FedEP                                                                                                                                              #
    #  1. Fitting the data distribution of each client's dataset to a Gaussian Mixture Model (GMM) with an Expectation-Maximization (EM) method.
    #  2. Sharing the parameters of the GMM as statistical characteristics of each neighbers.
    #  3. Using the shared statistical characteristics to calculate the global distribution.
    #  4. Calculating the cross entropy (KL divergence) between the global distribution and each client's local distribution
    #  5. Using these cross entropy to calculate the parameters for each node.    
    #  6. Using these parameters in model aggregation.
    ##########################################################################################################################################################
    
    def distribution_fitting_and_communication(self):
        """
        FedEP specific round of sharing distribution characteristics to learn the aggregation weights
        """

        fitting_thread = threading.Thread(target = self._distribution_fitting)
        print(f"[nodeFedEP] distribution fitting ......")
        fitting_thread.start()
        fitting_thread.join()
        print(f"[nodeFedEP] finished distribution fitting.")
        
        broadcast_thread = threading.Thread(target = self._distribution_broadcast)
        print(f"[nodeFedEP] broadcasting ......")
        broadcast_thread.start()
        broadcast_thread.join()
        print(f"[nodeFedEP] finished broadcast")

        if self.config.participant["device_args"]["role"] == Role.AGGREGATOR:
            self.aggregator.pooling(self._labels)


    def _distribution_broadcast(self):
        """
        FedEP specific round of broadcasting the distribution characteristics to the other nodes
        """
        self.aggregator._lock_to_start_pooling.acquire()
        # gets aggregator ready to accecpt distributions
        if self.config.participant["device_args"]["role"] == Role.AGGREGATOR:
            logging.info("[FedEP] Role.AGGREGATOR start waiting distribution...")
            print("[FedEP] Role.AGGREGATOR start waiting distribution...")
            self.aggregator.set_waiting_distribution(True)

        # get neighbors
        logging.info(f"[NodeFedEP] getting neighbors")
        neighbors = self._neighbors.get_all()
        logging.info(f"[NodeFedEP] neighbors: {neighbors}")
        print(f"[NodeFedEP] neighbors: {neighbors}")

        # broadcast distribution
        try:
            if self.config.participant["device_args"]["role"] != Role.IDLE:
                local_distribution = {"".join(self.addr): (self.theta, self.learner.get_num_samples()[0])}
                self.aggregator.add_distribution(
                    theta= self.theta,
                    addr= self.addr,
                    weight = self.learner.get_num_samples()[0],
                    all_neighbors = neighbors
                )
                logging.info(f"[NodeFedEP] Broadcasting distribution to {len(neighbors)} neighbors")
                for des in neighbors:
                    if des != self.addr:
                        self.send_distribution(des, local_distribution)
        except Exception as e:
            print(f"[NodeFedEP] Error broadcasting distribution: {e}")
    

    def send_distribution(self, des, distribution):
        try:
            print(f"[nodeFedEP] ({self.addr}) Sending distribution to {des}")
            
            # Initialize channel and stub
            channel = grpc.insecure_channel(des)
            stub = node_pb2_grpc.NodeServicesStub(channel)
            # Encode the distribution using the learner's encode_parameters method
            encoded_distribution = self.learner.encode_parameters(params=distribution)
            # Send the encoded distribution to the destination
            res = stub.add_distribution(
                node_pb2.Distributions(
                    source=self.addr,
                    distribution=encoded_distribution,
                ),
                timeout=10,
            )
            
            # Handling errors
            if res.error:
                print(f"[{self.addr}] Error while sending a model: {res.error}")

            # Close the channel after sending the distribution
            channel.close()

        except Exception as e:
            print(f"({self.addr}) Cannot send model to {des}. Error: {str(e)}")


    ############################
    #  GRPC - Remote Services  #
    ############################
    
    def add_distribution(self, request, context):
        """
        Adds a distribution to the aggregator
        """
        try:
            distribution = self.learner.decode_parameters(request.distribution)
            print(f"[NodeFedEP] Received distribution: {distribution}")
            received_theta, received_weight = next(iter(distribution.values()))
            self.aggregator.add_distribution(
                theta=received_theta,
                addr=[request.source],
                weight=received_weight,
                all_neighbors = self._neighbors.get_all()
            )
            return node_pb2.ResponseMessage(error="")
        except Exception as e:
            return node_pb2.ResponseMessage(error=str(e))




    def _distribution_fitting(self):
        '''
        FedEP specific round of fitting the distribution of the node with a GMMs model
        
        Args: Y: complete sorted of lable that is not unique, for example ([0,0,0,....,8,8,8,9,9,9])

        return:
            theta_hs: the parameters of the GMMs is a 3 x M matrix,  [π, μ, σ^2], where M is the number of mixture components.
                The mixture coefficient vector π = [π1, π2, . . . , πM ], with each element as the coefficient of the m-th Gaussian distribution.
                The vector μ = [μ1, μ2, . . . , μM ], with each element as the mean of the m-th Gaussian distribution. 
                The vector σ^2 = [σ^2_1 , σ^2_2 , . . . , σ^2_M ],with each element as the variance of the m-th Gaussian distribution. 
            likelihood: the likelihood of the data given theta_hs
        '''
        Y = np.array(np.sort(self._labels))
        # deciede the maximum number of mixture components
        Ms = math.ceil(len(set(Y.tolist())) * self._distribution_fitting_max_components_fraction) 
        theta_hs = np.empty(Ms, dtype=object)
        likelihood_hs = np.zeros(Ms)
        BICs = np.zeros(Ms)
        # AICs = np.zeros(Ms)
        for M in range(0,Ms):
            theta_hs[M], likelihood_hs[M] = self._expectation_maximum_algorithm(M+1 , Y)
            BICs[M] = -2*likelihood_hs[M] + M * np.log(len(Y))
            # AICs[M] = -2*likelihood_hs[M] + 2 * M
        min_BIC_index = np.argmin(BICs)
        print(f"[FedEP] local theta: {theta_hs[min_BIC_index]}")
        self.theta = theta_hs[min_BIC_index]

    
    def _expectation_maximum_algorithm(self, M, Y):
        '''
        derived theta by EM algorithm

        Args: M: the number of mixture components
              Y: complete list of lable that is ununique, for example ([0,0,0,....,8,8,8,9,9,9,])
        return:
            theta: the parameters of the GMMs given M and Y 
        '''
        theta = self._parameter_initialization(M,Y)
        likelihood_prev = 0
        theta_prev = theta
        iteration = 0
        while iteration <  self._EM_algorithm_max_iterations:
            gamma_lm, n_m = self._E_step(theta,Y)
            theta, likelihood = self._M_step(gamma_lm, n_m, Y)
            iteration += 1
            if likelihood == np.NINF or math.isnan(likelihood):
                return theta_prev, likelihood_prev
            if abs(likelihood - likelihood_prev) < self._EM_algorithm_epsilon:
                break
            likelihood_prev = likelihood
            theta_prev = theta
        return theta, likelihood

    def _parameter_initialization(self, M,Y):
        '''
        Initialized theta

        Args: M: the number of mixture components
              Y: complete list of lable that is ununique, for example ([0,0,0,....,8,8,8,9,9,9,])
        '''
        L=len(Y)
        # π (mixture weights)
        pi = np.random.rand(M)
        pi /= np.sum(pi)
        # μ (means)
        mu = Y[np.random.choice(L, M, replace=False)]
        # ϵ^2 (variances)
        sigma_squared = [np.var(Y.tolist())] * M
        # theta_0
        return np.column_stack((pi, mu, sigma_squared))
    
    def _gaussian(self, Y, mu, sigma_squared):
        '''
        probability density function of Gaussian distribution
        '''
        return np.exp(-np.square(Y-mu+self._gaussian_epsilon)/(2*(sigma_squared+self._gaussian_epsilon)))/(np.sqrt(2 * np.pi * (sigma_squared + self._gaussian_epsilon)))

    def _E_step(self, theta,Y):
        '''
        given theta, calculate the latent variable gamma_lm and the number of samples n_m for each mixture component m
        '''
        M = theta.shape[0]
        gamma_lm = np.zeros((len(Y),M))
        n_m = np.zeros(M)
        sum_gaussians = torch.zeros([len(Y)])
        for m in range(M):
            sum_gaussians += theta[m,0] * self._gaussian(Y, theta[m,1], theta[m,2])
        for m in range(M):
            gamma_lm[:,m] = theta[m,0] * self._gaussian(Y, theta[m,1], theta[m,2])/ sum_gaussians
            n_m[m] = np.sum(gamma_lm[:,m])
        return gamma_lm, n_m
    
    def _M_step(self, gamma_lm, n_m, Y):
        '''
        given gamma and n_m, calculate the new theta
        '''
        pi_h = n_m / len(Y)
        mu_h = np.array([gamma_lm[:,m] @ Y / n_m[m] for m in range(len(n_m))])
        sigma_squared_h = np.array([gamma_lm[:,m] @ ((Y-mu_h[m])**2 + self._gaussian_epsilon) / n_m[m] for m in range(len(n_m))])
        theta_h = np.column_stack((pi_h, mu_h, sigma_squared_h))

        likelihood = np.sum([n_m[m] * np.log(pi_h[m])+ gamma_lm[:,m] @ np.log(self._gaussian(Y, mu_h[m], sigma_squared_h[m])+self._gaussian_epsilon) for m in range(len(n_m))])
        return theta_h, likelihood
    
    







        
       
    

       
 