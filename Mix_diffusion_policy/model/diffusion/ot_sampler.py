import torch
import math
from torch.distributions import Beta
from torchcfm.conditional_flow_matching import ConditionalFlowMatcher
from torchcfm.optimal_transport import OTPlanSampler
from typing import Union
def pad_t_like_x(t, x):
    if isinstance(t, (float, int)):
        return t
    return t.reshape(-1, *([1] * (x.dim() - 1)))

class TransformedDistribution:
    def __init__(self, alpha=0.5, beta_1=2, beta_2=2, num_points=1000):
        self.alpha = alpha 
        self.beta_1 = beta_1
        self.beta_2 = beta_2
        self.x_values = torch.linspace(0, 1, num_points)
        self.pdf_values = self.custom_distribution(self.x_values)
        self.pdf_values /= torch.trapz(self.pdf_values, self.x_values)  # 归一化

    def custom_distribution(self, x):
        return self.alpha * (x ** self.beta_1) + (1 - self.alpha) * ((1 - x) ** self.beta_2)

    def sample(self, n_samples):
        uniform_samples = torch.rand(n_samples)
        cumulative_pdf = torch.cumsum(self.pdf_values, dim=0)
        cumulative_pdf_ = cumulative_pdf.clone()  # 创建副本
        cumulative_pdf /= cumulative_pdf_[-1]  # 归一化
        indices = torch.searchsorted(cumulative_pdf, uniform_samples)
        sampled_x = self.x_values[indices]
        sampled_x[uniform_samples == 1] = 1.0
        return sampled_x

class OTVariancePreservingConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma: Union[float, int] = 0.0):
        super().__init__(sigma)
        self.ot_sampler = OTPlanSampler(method="exact")
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        x0, x1 = self.ot_sampler.sample_plan(x0, x1)
        return super().sample_location_and_conditional_flow(x0, x1, t, return_noise)
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return torch.cos(math.pi / 2 * t) * x0 + torch.sin(math.pi / 2 * t) * x1

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * (torch.cos(math.pi / 2 * t) * x1 - torch.sin(math.pi / 2 * t) * x0)
    
class LogiticSquaredVPConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return x0*torch.cos(math.pi / 2 * t)**2 + x1*torch.sin(math.pi / 2 * t)**2 

    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        t = pad_t_like_x(t, x0)
        return math.pi / 2 * torch.sin(math.pi * t)*(x1-x0)
        # return math.pi * (torch.cos(math.pi / 2 * t)*torch.sin(math.pi / 2 * t) * x1 - torch.cos(math.pi / 2 * t)*torch.sin(math.pi / 2 * t) * x0)
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

class LogiticGVPConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, alpha,beta,sigma=0.0):
        super().__init__(sigma)
        self.alpha_1=alpha
        self.beta_1=beta
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return x0*torch.cos(math.pi/2 * t)**self.alpha_1 + x1*torch.sin(math.pi / 2 * t)**self.beta_1
    
    def compute_conditional_flow(self, x0, x1, t, xt):
        del xt
        t = pad_t_like_x(t, x0)
        d_alpha_t= -math.pi / 2 *self.alpha_1*(torch.cos(math.pi / 2 * t))**(self.alpha_1-1) * torch.sin(math.pi / 2 * t)
        d_beta_t= math.pi / 2 *self.beta_1*(torch.sin(math.pi / 2 * t))**(self.beta_1-1) * torch.cos(math.pi / 2 * t)
        return d_alpha_t*x0+d_beta_t*x1

    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
        
class LogiticConditionalFlowMatcher(ConditionalFlowMatcher):
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, xt)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

class SBConditionalFlowMatcher(ConditionalFlowMatcher):
    def compute_sigma_t(self, t):
        return self.sigma * torch.sqrt(t * (1 - t))
    def compute_conditional_flow(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * t * (1 - t) + 1e-8)
        ut = sigma_t_prime_over_sigma_t * (xt - mu_t) + x1 - x0
        return ut
 
class GVPStochasticConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, alpha,beta,sigma=0.0):
        super().__init__(sigma)
        self.alpha_1=alpha
        self.beta_1=beta
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return x0*torch.cos(math.pi/2 * t)**self.alpha_1 + x1*torch.sin(math.pi / 2 * t)**self.beta_1
    
    def compute_mu_prime_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        d_alpha_t= -math.pi / 2 *self.alpha_1*(torch.cos(math.pi / 2 * t))**(self.alpha_1-1) * torch.sin(math.pi / 2 * t)
        d_beta_t= math.pi / 2 *self.beta_1*(torch.sin(math.pi / 2 * t))**(self.beta_1-1) * torch.cos(math.pi / 2 * t)
        return d_alpha_t*x0+d_beta_t*x1
    
    def compute_sigma_t(self, t):
        # return self.sigma * torch.sqrt(t * (1 - t))
        return self.sigma * torch.sqrt(1 - t)
    def compute_log_t(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        return (mu_t-xt)/(t*(1-t)*self.sigma**2+1e-8)
    
    def compute_conditional_flow(self, x0, x1, t, epsilon):
        t = pad_t_like_x(t, x0)
        # sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * torch.sqrt(t * (1 - t)) + 1e-8)
        sigma_t_prime_over_sigma_t = - 1 / (2 * torch.sqrt(1 - t) + 1e-8)
        mu_prime_t=self.compute_mu_prime_t(x0, x1, t)
        ut = sigma_t_prime_over_sigma_t * self.sigma * epsilon + mu_prime_t
        return ut
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

class GVPNoiseStochasticConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, alpha,beta,sigma_z=1.5,sigma=0.0):
        super().__init__(sigma)
        self.alpha_1=alpha
        self.beta_1=beta
        self.sigma_z=sigma_z
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return x0*torch.cos(math.pi/2 * t)**self.alpha_1 + x1*torch.sin(math.pi / 2 * t)**self.beta_1
    
    def compute_mu_prime_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        d_alpha_t= -math.pi / 2 *self.alpha_1*(torch.cos(math.pi / 2 * t))**(self.alpha_1-1) * torch.sin(math.pi / 2 * t)
        d_beta_t= math.pi / 2 *self.beta_1*(torch.sin(math.pi / 2 * t))**(self.beta_1-1) * torch.cos(math.pi / 2 * t)
        return d_alpha_t*x0+d_beta_t*x1
    
    def compute_sigma_t(self, t):
        return self.sigma * (1 - t)**self.sigma_z

    def compute_conditional_flow(self, x0, x1, t, epsilon):
        t = pad_t_like_x(t, x0)
        # sigma_t_prime_over_sigma_t = (1 - 2 * t) / (2 * torch.sqrt(t * (1 - t)) + 1e-8)
        sigma_t_prime_over_sigma_t = - self.sigma_z*(1-t)**(self.sigma_z-1)

        mu_prime_t=self.compute_mu_prime_t(x0, x1, t)
        ut = sigma_t_prime_over_sigma_t * self.sigma * epsilon + mu_prime_t
        return ut
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

class BetaConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self,sigma=0.0):
        super().__init__(sigma)
        self.beta_dis=Beta(1.5,1)
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = self.beta_dis.sample(sample_shape=(x0.shape[0],)).type_as(x0)
            t = 0.999*(1-u)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut
class BetaStochasticConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, sigma_z=2.0,sigma=0.0):
        super().__init__(sigma)
        self.sigma_z=sigma_z
        self.beta_dis=Beta(1.5,1)
    
    def compute_sigma_t(self, t):
        return self.sigma * (1 - t)**self.sigma_z
    
    def compute_conditional_flow(self, x0, x1, t, epsilon):
        t = pad_t_like_x(t, x0)
        sigma_t_prime_over_sigma_t = - self.sigma_z*(1-t)**(self.sigma_z-1)
        mu_prime_t=x1-x0
        ut = sigma_t_prime_over_sigma_t * self.sigma * epsilon + mu_prime_t
        return ut
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            u = self.beta_dis.sample(sample_shape=(x0.shape[0],)).type_as(x0)
            t = 1-u
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut



class CustomConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self,sigma=0.0):
        super().__init__(sigma)
        self.dis=TransformedDistribution()
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_noise=False):
        if t is None:
            t=self.dis.sample(x0.shape[0]).type_as(x0)
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        if return_noise:
            return t, xt, ut, eps
        else:
            return t, xt, ut

class GVPNSDEConditionalFlowMatcher(ConditionalFlowMatcher):
    def __init__(self, alpha,beta,sigma=0.0):
        super().__init__(sigma)
        self.alpha_1=alpha
        self.beta_1=beta
    def compute_mu_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        return x0*torch.cos(math.pi/2 * t)**self.alpha_1 + x1*torch.sin(math.pi / 2 * t)**self.beta_1
    
    def compute_mu_prime_t(self, x0, x1, t):
        t = pad_t_like_x(t, x0)
        d_alpha_t= -math.pi / 2 *self.alpha_1*(torch.cos(math.pi / 2 * t))**(self.alpha_1-1) * torch.sin(math.pi / 2 * t)
        d_beta_t= math.pi / 2 *self.beta_1*(torch.sin(math.pi / 2 * t))**(self.beta_1-1) * torch.cos(math.pi / 2 * t)
        return d_alpha_t*x0+d_beta_t*x1
    
    def compute_sigma_t(self, t):
        return self.sigma * (1 - t)

    def compute_conditional_flow(self, x0, x1, t, epsilon):
        t = pad_t_like_x(t, x0)
        sigma_t_prime_over_sigma_t = - 1
        mu_prime_t=self.compute_mu_prime_t(x0, x1, t)
        ut = sigma_t_prime_over_sigma_t * self.sigma * epsilon + mu_prime_t
        return ut
    
    def compute_log_t(self, x0, x1, t, xt):
        t = pad_t_like_x(t, x0)
        mu_t = self.compute_mu_t(x0, x1, t)
        var = self.compute_sigma_t(t)
        return -(xt-mu_t)/(var.clamp_min(1e-5))
    
    def sample_location_and_conditional_flow(self, x0, x1, t=None, return_score=False):
        if t is None:
            u = torch.normal(0.0, 1.0, size=(x0.shape[0],)).type_as(x0)
            t = 1 / (1 + torch.exp(-u))
        assert len(t) == x0.shape[0], "t has to have batch size dimension"
        eps = self.sample_noise_like(x0)
        xt = self.sample_xt(x0, x1, t, eps)
        ut = self.compute_conditional_flow(x0, x1, t, eps)
        logt=self.compute_log_t(x0,x1,t,xt)
        if return_score:
            return t, xt, ut, logt
        else:
            return t, xt, ut