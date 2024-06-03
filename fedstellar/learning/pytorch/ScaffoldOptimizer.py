import torch


class ScaffoldOptimizer(torch.optim.Adam):
    """
    Custom optimizer that modifies the update step to include a control variable c.
    """
    def __init__(self, params, lr=1e-3, c = None, ci=None):
        """
        Initializes the optimizer.
        """
        super(ScaffoldOptimizer, self).__init__(params, lr=lr)
        self.ci = ci
        self.c = c

    def initialize_c(self, grad):
        """
        updates the control variable c.
        """
        self.ci = [torch.zeros_like(grad)]
        self.c = [torch.zeros_like(grad)]

    def update_c(self,new_c):
        self.c.append(new_c)

    def update_ci(self,new_ci):
        self.ci.append(new_ci)

    def step(self, closure=None):
        """
        Performs a single optimization step.
        """
        # Compute gradients
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            print(f"[optimizer] group:{group}")
            for p, in group['params']:
                print(f"[optimizer] p:{p}")
                if p.grad is None:
                    continue
                grad = p.grad.data
                if self.ci == None:
                    self.initialize_c(grad)
                # Add the constant to gradients
                c_avg = torch.mean(torch.stack(self.c))
                ci_avg = torch.mean(torch.stack(self.ci))
                dp = p.grad.data + c_avg - ci_avg
                p.data = p.data - dp.data * group['lr']

        return loss