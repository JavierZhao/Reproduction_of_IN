import itertools
import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

class GraphNet(nn.Module):
    def __init__(self, n_constituents, n_targets, params, hidden, n_vertices, params_v, vv_branch=False, De=5, Do=6, softmax=False):
        super(GraphNet, self).__init__()
        self.hidden = int(hidden)
        # Feature vector of length P
        self.P = params
        # Np particles each represented by feature vector P
        self.N = n_constituents
        # Feature vectof of length S
        self.S = params_v
        self.Nv = n_vertices
        # Connecting each particle to every other particle, Npp = Np(Np-1)
        self.Nr = self.N * (self.N - 1)
        # Npv = Np Nv
        self.Nt = self.N * self.Nv
        self.Ns = self.Nv * (self.Nv - 1)
        self.Dr = 0
        # De is the size of internal representation
        self.De = De
        self.Dx = 0
        # used for size of matrix O
        self.Do = Do
        self.n_targets = n_targets
        self.assign_matrices()
        self.assign_matrices_SV()
        self.vv_branch = vv_branch
        self.softmax = softmax
        # used for vertex-vertex branch
        # keep for now, but I think we only use vertex-particle branch
        if self.vv_branch:
            self.assign_matrices_SVSV()
        # interaction functions setup for Epp matrix
        self.fr1 = nn.Linear(2 * self.P + self.Dr, self.hidden).cuda()
        self.fr2 = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fr3 = nn.Linear(int(self.hidden), self.De).cuda()
        # interaction functions setup for Evp matrix
        self.fr1_pv = nn.Linear(self.S + self.P + self.Dr, self.hidden).cuda()
        self.fr2_pv = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fr3_pv = nn.Linear(int(self.hidden), self.De).cuda()
        if self.vv_branch:
            self.fr1_vv = nn.Linear(2 * self.S + self.Dr, self.hidden).cuda()
            self.fr2_vv = nn.Linear(self.hidden, int(self.hidden)).cuda()
            self.fr3_vv = nn.Linear(int(self.hidden), self.De).cuda()

        # interaction functions setup for O matrix
        self.fo1 = nn.Linear(self.P + self.Dx + (2 * self.De), self.hidden).cuda()
        self.fo2 = nn.Linear(self.hidden, int(self.hidden)).cuda()
        self.fo3 = nn.Linear(int(self.hidden), self.Do).cuda()
        if self.vv_branch:
            self.fo1_v = nn.Linear(self.S + self.Dx + (2 * self.De), self.hidden).cuda()
            self.fo2_v = nn.Linear(self.hidden, int(self.hidden)).cuda()
            self.fo3_v = nn.Linear(int(self.hidden), self.Do).cuda()
        if self.vv_branch:
            self.fc_fixed = nn.Linear(2*self.Do, self.n_targets).cuda()
        else:
            self.fc_fixed = nn.Linear(self.Do, self.n_targets).cuda()

        # Setup Rr and Rs matrices
    def assign_matrices(self):
        # initialize receiving matrix (Rr)
        self.Rr = torch.zeros(self.N, self.Nr)
        # initialize sending matrix (Rs)
        self.Rs = torch.zeros(self.N, self.Nr)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.N)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            # Element is set to 1 when the ith particle recieves/sends jth edge and zero otherwise
            # same for both Rr and Rs
            self.Rr[r, i] = 1
            self.Rs[s, i] = 1
        self.Rr = (self.Rr).cuda()
        self.Rs = (self.Rs).cuda()

    # Setup Rk and Rv matrices
    def assign_matrices_SV(self):
        # initialize adjacency matrices Rk and Rv
        self.Rk = torch.zeros(self.N, self.Nt)
        self.Rv = torch.zeros(self.Nv, self.Nt)
        receiver_sender_list = [i for i in itertools.product(range(self.N), range(self.Nv))]
        for i, (k, v) in enumerate(receiver_sender_list):
            self.Rk[k, i] = 1
            self.Rv[v, i] = 1
        self.Rk = (self.Rk).cuda()
        self.Rv = (self.Rv).cuda()

    ## Don't think we use this... ##
    def assign_matrices_SVSV(self):
        self.Rl = torch.zeros(self.Nv, self.Ns)
        self.Ru = torch.zeros(self.Nv, self.Ns)
        receiver_sender_list = [i for i in itertools.product(range(self.Nv), range(self.Nv)) if i[0]!=i[1]]
        for i, (l, u) in enumerate(receiver_sender_list):
            self.Rl[l, i] = 1
            self.Ru[u, i] = 1
        self.Rl = (self.Rl).cuda()
        self.Ru = (self.Ru).cuda()
    ###############################

    # creates particle-particle interaction matrix Bpp Dim(2*P, Npp)
    # leading to effect matrix Epp Dim(DE, Npp)
    # then propagate the particle-particle interactions back to the particles receiving them:
    # Ebar_pp Dim(DE, Np)
    # Here x and y represent particle feature matrix and vertex feature matrix, respectively.
    def forward(self, x, y):
        ###PF Candidate - PF Candidate###
        Orr = self.tmul(x, self.Rr)
        Ors = self.tmul(x, self.Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        # processing each column of Bpp by the function f^{pp}_R
        B = nn.functional.relu(self.fr1(B.view(-1, 2 * self.P + self.Dr)))
        B = nn.functional.relu(self.fr2(B))
        E = nn.functional.relu(self.fr3(B).view(-1, self.Nr, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        # matrix multiplication to create Ebar_pp
        Ebar_pp = self.tmul(E, torch.transpose(self.Rr, 0, 1).contiguous())
        del E
        
        ####Secondary Vertex - PF Candidate### 
        # similar process as above, but for Ebar_vp
        Ork = self.tmul(x, self.Rk)
        Orv = self.tmul(y, self.Rv)
        B = torch.cat([Ork, Orv], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(self.fr1_pv(B.view(-1, self.S + self.P + self.Dr)))
        B = nn.functional.relu(self.fr2_pv(B))
        E = nn.functional.relu(self.fr3_pv(B).view(-1, self.Nt, self.De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar_pv = self.tmul(E, torch.transpose(self.Rk, 0, 1).contiguous())
        Ebar_vp = self.tmul(E, torch.transpose(self.Rv, 0, 1).contiguous())
        del E

        ###Secondary vertex - secondary vertex###
        ## Don't think this is used...##
        if self.vv_branch:
            Orl = self.tmul(y, self.Rl)
            Oru = self.tmul(y, self.Ru)
            B = torch.cat([Orl, Oru], 1)
            ### First MLP ###
            B = torch.transpose(B, 1, 2).contiguous()
            B = nn.functional.relu(self.fr1_vv(B.view(-1, 2 * self.S + self.Dr)))
            B = nn.functional.relu(self.fr2_vv(B))
            E = nn.functional.relu(self.fr3_vv(B).view(-1, self.Ns, self.De))
            del B
            E = torch.transpose(E, 1, 2).contiguous()
            Ebar_vv = self.tmul(E, torch.transpose(self.Rl, 0, 1).contiguous())
            del E
        #############################

        ####Final output matrix for particles###
        # C matrix with dimension (P + 2*DE) x Np
        # combines input information of each particle X with learned
        # representation of the particle-particle (Ebar_pp) and particle-vertex (Ebar_vp) interactions
        C = torch.cat([x, Ebar_pp, Ebar_pv], 1)
        del Ebar_pp
        del Ebar_pv
        C = torch.transpose(C, 1, 2).contiguous()
        ### Second MLP ###
        # final aggregator combines the input and interaction information
        # the aggregator is the function fo
        C = nn.functional.relu(self.fo1(C.view(-1, self.P + self.Dx + (2 * self.De))))
        C = nn.functional.relu(self.fo2(C))
        # creates postinteraction representation of the graph (Matrix O)
        O = nn.functional.relu(self.fo3(C).view(-1, self.N, self.Do))
        del C

        if self.vv_branch:
            ####Final output matrix for particles### 
            C = torch.cat([y, Ebar_vv, Ebar_vp], 1)
            del Ebar_vv
            del Ebar_vp
            C = torch.transpose(C, 1, 2).contiguous()
            ### Second MLP ###
            C = nn.functional.relu(self.fo1_v(C.view(-1, self.S + self.Dx + (2 * self.De))))
            C = nn.functional.relu(self.fo2_v(C))
            O_v = nn.functional.relu(self.fo3_v(C).view(-1, self.Nv, self.Do))
            del C
        
        #Taking the sum of over each particle/vertex
        # output of the classifier
        N = torch.sum(O, dim=1)
        del O
        if self.vv_branch:
            N_v = torch.sum(O_v,dim=1)
            del O_v
        
        ### Classification MLP ###
        if self.vv_branch:
            #N = nn.functional.relu(self.fc_1(torch.cat([N, N_v],1)))
            #N = nn.functional.relu(self.fc_2(N))
            #N = self.fc_3(N)
            N =self.fc_fixed(torch.cat([N, N_v],1))
        else:
            N = self.fc_fixed(N)

        if self.softmax:
            N = nn.Softmax(dim=-1)(N)

        return N 
            
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])