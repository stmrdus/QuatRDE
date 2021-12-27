import torch
import torch.nn as nn
import numpy as np
from .Model import Model
from numpy.random import RandomState

class QuatRDE(Model):
    def __init__(self, config):
        super(QuatRDE, self).__init__(config)

        self.ent = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.ent_transfer = nn.Embedding(self.config.entTotal, self.config.hidden_size * 4)
        self.rel_transfer = nn.Embedding(self.config.relTotal, self.config.hidden_size * 4)
        self.Whr = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        self.Wtr = nn.Embedding(self.config.relTotal, 4 * self.config.hidden_size)
        self.criterion = nn.Softplus()
        self.f = nn.PReLU()
        self.quaternion_init = False
        self.init_weights()

    def init_weights(self):
        nn.init.kaiming_uniform_(self.ent.weight.data)
        nn.init.kaiming_uniform_(self.rel.weight.data)
        nn.init.kaiming_uniform_(self.Whr.weight.data)
        nn.init.kaiming_uniform_(self.Wtr.weight.data)

    def _calc(self, h, r):
        s_a, x_a, y_a, z_a = torch.chunk(h, 4, dim=1)
        s_b, x_b, y_b, z_b = torch.chunk(r, 4, dim=1)

        denominator_b = torch.sqrt(s_b ** 2 + x_b ** 2 + y_b ** 2 + z_b ** 2)
        s_b = s_b / denominator_b
        x_b = x_b / denominator_b
        y_b = y_b / denominator_b
        z_b = z_b / denominator_b

        A = s_a * s_b - x_a * x_b - y_a * y_b - z_a * z_b # qrpr−qipi−qjpj−qkpk
        B = s_a * x_b + s_b * x_a + y_a * z_b - y_b * z_a 
        C = s_a * y_b + s_b * y_a + z_a * x_b - z_b * x_a
        D = s_a * z_b + s_b * z_a + x_a * y_b - x_b * y_a

        return torch.cat([A, B, C, D], dim=1)

    def loss(self, score, regul):
        return (
            torch.mean(self.criterion(score * self.batch_y)) + self.config.lmbda * regul
        )

    def regulation(self, x):
        a, b, c, d = torch.chunk(x, 4, dim=1)
        score = torch.mean(a ** 2) + torch.mean(b ** 2) + torch.mean(c ** 2) + torch.mean(d ** 2)
        return score
    def _transfer(self, x, x_transfer, r_transfer):
        ent_transfer = self._calc(x, x_transfer)
        ent_rel_transfer = self._calc(ent_transfer, r_transfer)
        return ent_rel_transfer

    def forward(self):
        # (h, r, t) embedding
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)

        # (h, r, t) transfer vector
        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)
        
        hr = self.Whr(self.batch_r)
        tr = self.Wtr(self.batch_r)

        # multiplication as QuatE
        h_r = self._calc(h1, hr)
        t_r = self._calc(t1, tr)
        hrr = self._calc(h_r, r)
        
        # Inner product as QuatE
        score = self.f(torch.sum(hrr * t_r, -1))

        regul = self.regulation(h) + self.regulation(r) + self.regulation(t) + \
                self.regulation(h_transfer) + self.regulation(r_transfer) + self.regulation(t_transfer) + \
                    self.regulation(hr) + self.regulation(tr)

        return self.loss(score, regul)

    def predict(self):
        h = self.ent(self.batch_h)
        r = self.rel(self.batch_r)
        t = self.ent(self.batch_t)

        h_transfer = self.ent_transfer(self.batch_h)
        t_transfer = self.ent_transfer(self.batch_t)
        r_transfer = self.rel_transfer(self.batch_r)
        h1 = self._transfer(h, h_transfer, r_transfer)
        t1 = self._transfer(t, t_transfer, r_transfer)

        hr = self.Whr(self.batch_r)
        tr = self.Wtr(self.batch_r)

        # multiplication as QuatE
        h_r = self._calc(h1, hr)
        t_r = self._calc(t1, tr)
        hrr = self._calc(h_r, r)
        
        # Inner product as QuatE
        score = self.f(torch.sum(hrr * t_r, -1))

        return score.cpu().data.numpy()