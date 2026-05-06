import torch
import torch.distributions as D

class customProb2D:
    def __init__(self, loc, var):
        self.prob1 = D.Normal(loc + torch.tensor([-12.0, 0.0]), var)
        self.prob2 = D.Normal(loc + torch.tensor([0.0, -12.0]), var)
        self.prob3 = D.Normal(loc, var)
        self.prob4 = D.Normal(loc + torch.tensor([0.0, 12.0]), var)
        self.prob5 = D.Normal(loc + torch.tensor([12.0, 0.0]), var)
        
    def sample(self, N) -> torch.Tensor:
        N = int(N/5)
        x1 = self.prob1.sample((N, ))
        x2 = self.prob2.sample((N, ))
        x3 = self.prob3.sample((N, ))
        x4 = self.prob4.sample((N, ))
        x5 = self.prob5.sample((N, ))
        x = torch.concat((x1, x2, x3, x4, x5), dim=0)
        return x