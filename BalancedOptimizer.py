import torch
class BalancedOptimizer(torch.optim.Optimizer):
    def __init__(self, params, lr, weight_decay):
        self.gradients = {"balanced": {}, "train": {}}
        self.weight_decay = weight_decay
        self.lr = lr
        defaults = dict(lr=lr, weight_decay=weight_decay)
        super(BalancedOptimizer, self).__init__(params, defaults)
    def store_gradients(self, name):
        params_with_grad_total = []
        d_p_list_total = []
        for group in self.param_groups:
            # print(group)
            # Store all parameters with gradients, and their corresponding gradients
            params_with_grad = []
            d_p_list = []
            # Loop through each parameter
            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    d_p_list.append(p.grad)
            params_with_grad_total.append(params_with_grad)
            d_p_list_total.append(d_p_list)
        self.gradients[name]["params"] = params_with_grad_total
        self.gradients[name]["d_p"] = d_p_list_total

    def step(self, closure=None):
        # Loop through each param group
        for params_with_grad, d_p_list in zip(self.gradients["balanced"]["params"], self.gradients["balanced"]["d_p"]):
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                with torch.no_grad():
                    if self.weight_decay != 0:
                        d_p = d_p.add(param, alpha=self.weight_decay)
                    param.add_(d_p, alpha=-self.lr/2)
        for params_with_grad, d_p_list in zip(self.gradients["train"]["params"], self.gradients["train"]["d_p"]):
            for i, param in enumerate(params_with_grad):
                d_p = d_p_list[i]
                with torch.no_grad():
                    param.add_(d_p, alpha=-self.lr/2)

        self.gradients["balanced"]["params"] = []
        self.gradients["balanced"]["d_p"] = []
        self.gradients["train"]["params"] = []
        self.gradients["train"]["d_p"] = []