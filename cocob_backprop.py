import numpy as np
import cntk as C
import math
np.set_printoptions(precision=4)

class COCOBBackprop(C.UserLearner):
    """COCOB-Backprop optimization algorithm.
    See: https://arxiv.org/abs/1705.07795
    """
    
    def __init__(self, alpha=100, eps=1e-8):
        """
        Initialize algorithm
        :param alpha: int, number of vocabulary
        :param eps: float, Starting amount of money 
        """
        self._alpha = alpha
        self.eps = eps
        defaults = dict(alpha=alpha, eps=eps)
        super(COCOBBackprop, self).__init__(params, defaults)
        
    def step(self, closure=None):
        """
        Initialize algorithm
        :param closure: dict
        :return loss: float 
        """
        loss = None
        
        if closure is not None:
            loss = closure()
            
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
        
                grad = p.grad.data
                state = self.state[p]
                
                if len(state) == 0:
                    state['gradients_sum'] = np.zeros_like(p.data).float()
                    state['grad_norm_sum'] = np.zeros_like(p.data).float()
                    state['L'] = self.eps * np.ones_like(p.data).float()
                    state['tilde_w'] = np.zeros_like(p.data).float()
                    state['reward'] = np.zeros_like(p.data).float()
                    
                gradients_sum = state['gradients_sum']
                grad_norm_sum = state['grad_norm_sum']
                tilde_w = state['tilde_w']
                L = state['L']
                reward = state['reward']
                
                zero = np.float32([0.])
                
                L_update = np.max(L, np.abs(grad))
                gradients_sum_update = gradients_sum + grad
                grad_norm_sum_update = grad_norm_sum + np.abs(grad)
                reward_update = np.max(reward - grad * tilde_w, zero)
                new_w = -gradients_sum_update/(L_update * (np.max(grad_norm_sum_update + L_update, self._alpha * L_update)))*(reward_update + L_update)
                p.data = p.data - tilde_w + new_w
                tilde_w_update = new_w
                
                state['gradients_sum'] = gradients_sum_update
                state['grad_norm_sum'] = grad_norm_sum_update
                state['L'] = L_update
                state['tilde_w'] = tilde_w_update
                state['reward'] = reward_update

        return loss
