import torch

class PurePythonAdam:
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8):
        self.params = list(params)
        self.param_groups = [{'params': self.params}]
        self.lr = lr
        self.beta1, self.beta2 = betas
        self.eps = eps
        self.m = [torch.zeros_like(p.data) for p in self.params]
        self.v = [torch.zeros_like(p.data) for p in self.params]
        self.t = 0
        
    def step(self, closure=None):
        self.t += 1
        with torch.no_grad():
            for i, p in enumerate(self.params):
                if p.grad is None:
                    continue
                grad = p.grad
                
                # m = beta1 * m + (1 - beta1) * grad
                self.m[i].mul_(self.beta1).add_(grad, alpha=1 - self.beta1)
                
                # v = beta2 * v + (1 - beta2) * (grad ** 2)
                self.v[i].mul_(self.beta2).addcmul_(grad, grad, value=1 - self.beta2)
                
                m_hat = self.m[i] / (1 - self.beta1 ** self.t)
                v_hat = self.v[i] / (1 - self.beta2 ** self.t)
                
                p.data.addcdiv_(m_hat, v_hat.sqrt() + self.eps, value=-self.lr)

    def zero_grad(self, set_to_none=False):
        for p in self.params:
            if p.grad is not None:
                if set_to_none:
                    p.grad = None
                else:
                    if p.grad.grad_fn is not None:
                        p.grad.detach_()
                    else:
                        p.grad.requires_grad_(False)
                    p.grad.zero_()

    def state_dict(self):
        return {'m': self.m, 'v': self.v, 't': self.t, 'lr': self.lr, 'beta1': self.beta1, 'beta2': self.beta2, 'eps': self.eps}

    def load_state_dict(self, state):
        self.m = state['m']
        self.v = state['v']
        self.t = state['t']
        self.lr = state['lr']
        self.beta1 = state['beta1']
        self.beta2 = state['beta2']
        self.eps = state['eps']
