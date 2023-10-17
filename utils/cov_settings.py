import torch

def make_cov_index_matrix(dim):  # make symmetric matrix for embedding index
    matrix = torch.LongTensor()
    s_index = 0
    for i in range(dim):
        matrix = torch.cat([matrix, torch.arange(s_index, s_index + dim).unsqueeze(0)], dim=0)
        s_index += (dim - (2 + i))
    return matrix.triu(diagonal=1).transpose(0, 1) + matrix.triu(diagonal=1)


class CovMatrix_AIAW_real:
    def __init__(self, dim, relax_denom=0):
        """
            dim (int): The dimension of the covariance matrix.
            relax_denom (int, optional): A parameter that controls the relaxation of the number of off-diagonal entries to consider. Defaults to 0.
        """
        super(CovMatrix_AIAW_real, self).__init__()
        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.cov_matrix = None
        self.count_pair_cov = 0
        self.mask_matrix = None
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        # Returns the identity matrix and its transposed version, used for calculations.
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        #  returns the identity matrix, the mask matrix, the number of sensitive entries, and the number of removed covariances.
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None


    def set_mask_matrix(self):
        # torch.set_printoptions(threshold=500000)
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)

        if self.margin == 0:    
            num_sensitive = int(3/1000 * cov_flatten.size()[0])
            print('cov_flatten.size()[0]', cov_flatten.size()[0])
            print("num_sensitive =", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)

    def set_pair_covariance(self, pair_cov):
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            self.cov_matrix = self.cov_matrix + pair_cov
        self.count_pair_cov += 1


class CovMatrix_AIAW_fake:
    def __init__(self, dim, relax_denom=0):
        super(CovMatrix_AIAW_fake, self).__init__()

        self.dim = dim
        self.i = torch.eye(dim, dim).cuda()

        # print(torch.ones(16, 16).triu(diagonal=1))
        self.reversal_i = torch.ones(dim, dim).triu(diagonal=1).cuda()

        # num_off_diagonal = ((dim * dim - dim) // 2)  # number of off-diagonal
        self.num_off_diagonal = torch.sum(self.reversal_i)
        self.num_sensitive = 0
        self.cov_matrix = None
        self.count_pair_cov = 0
        self.mask_matrix = None
        print("num_off_diagonal", self.num_off_diagonal)
        if relax_denom == 0:
            print("relax_denom == 0!!!!!")
            self.margin = 0
        else:                   # do not use
            self.margin = self.num_off_diagonal // relax_denom

    def get_eye_matrix(self):
        return self.i, self.reversal_i

    def get_mask_matrix(self, mask=True):
        if self.mask_matrix is None:
            self.set_mask_matrix()
        return self.i, self.mask_matrix, 0, self.num_sensitive

    def reset_mask_matrix(self):
        self.mask_matrix = None

    def set_mask_matrix(self):
        # torch.set_printoptions(threshold=500000)
        self.cov_matrix = self.cov_matrix / self.count_pair_cov
        cov_flatten = torch.flatten(self.cov_matrix)

        if self.margin == 0:    
            num_sensitive = int(0.6/1000 * cov_flatten.size()[0])
            print('cov_flatten.size()[0]', cov_flatten.size()[0])
            print("num_sensitive =", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        else:                   # do not use
            num_sensitive = self.num_off_diagonal - self.margin
            print("num_sensitive = ", num_sensitive)
            _, indices = torch.topk(cov_flatten, k=int(num_sensitive))
        mask_matrix = torch.flatten(torch.zeros(self.dim, self.dim).cuda())
        mask_matrix[indices] = 1

        if self.mask_matrix is not None:
            self.mask_matrix = (self.mask_matrix.int() & mask_matrix.view(self.dim, self.dim).int()).float()
        else:
            self.mask_matrix = mask_matrix.view(self.dim, self.dim)
        self.num_sensitive = torch.sum(self.mask_matrix)
        print("Check whether two ints are same", num_sensitive, self.num_sensitive)

        self.var_matrix = None
        self.count_var_cov = 0

        if torch.cuda.current_device() == 0:
            print("Covariance Info: (CXC Shape, Num_Off_Diagonal)", self.mask_matrix.shape, self.num_off_diagonal)
            print("Selective (Sensitive Covariance)", self.num_sensitive)

    def set_pair_covariance(self, pair_cov):
        if self.cov_matrix is None:
            self.cov_matrix = pair_cov
        else:
            self.cov_matrix = self.cov_matrix + pair_cov
        self.count_pair_cov += 1
