from .lightninglearner import LightningLearner

class LightningLearnerSCAFFOLD(LightningLearner):
    def __init__(self, model, data, config=None, logger=None):
        super().__init__(model, data, config, logger)

    
    def model_update_c(self, c):
        """
        Updates the control variable c.
        """
        self.model.update_c(c)

    def get_client_controls(self):
        return self.model.get_client_controls()
    
    # def update_ci(self, ci):
    #     self.model.update_ci(ci)
         

