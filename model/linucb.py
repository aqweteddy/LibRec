from typing import List

import torch


class LinUCB:
    """
    LinUCB algorithm implementation
    """

    def __init__(self, num_books, book_feat_size, user_feat_size, alpha, device='cpu'):
        """
        Parameters
        ----------
        alpha : number
            LinUCB parameter
        """  
        self.device = device
        
        d = book_feat_size + user_feat_size # size for A, b matrices: num of features for articles + num of features for user
        self.A = torch.stack([torch.eye(d)] * num_books)
        self.b = torch.zeros((num_books, d)).to(self.device)
        self.d = d
        self.alpha = round(alpha, 1)
        self.feat_size = (book_feat_size, user_feat_size)
        self.algorithm = "LinUCB (α=" + str(self.alpha) + ")"

        print(f"A shape: {self.A.shape}\nb shape: {self.b.shape}\nalpha: {self.alpha}")
        
    def predict(self, user, cand_index, cand_feat):
        """
         Returns the best arm's index relative to the pool
         Parameters
         ----------
         t : number
             number of trial
         user : array
             user features
         pool_idx : array of indexes
             pool indexes for article identification
         """
        A = self.A[cand_index].to(self.device)  # [cand_size, d, d]
        b = self.b[cand_index].unsqueeze(-1)  #  [cand_size, d, 1]
        
        user = user.repeat(len(cand_index), 1) # [cand_size, ufeat_size]

        x = torch.cat((user, cand_feat), dim=-1) # [cand_size, d]
        x = x.unsqueeze(-1)  # [cand_size, d, 1]
        
        A = torch.linalg.pinv(A)
        theta = A @ b  # [cand_size, d, 1]
        print(A.shape, theta.shape, b.shape, x.shape)
        p = theta.permute(0, 2, 1) @ x
        p += self.alpha * torch.sqrt(x.permute(0, 2, 1)) @ A @ x
        p = p.squeeze(-1).squeeze(-1)
        result = torch.argsort(p, dim=-1, descending=True)
        return result
    
    def train_one_records(self, stu_feat: torch.Tensor, 
                          cand_books_feat: torch.Tensor,
                          cand_books_index: List[int], # in parameter
                          ground_book_label: int # in cand_books_feat
                          ):
        stu_feat = stu_feat.to(self.device)
        cand_books_feat = cand_books_feat.to(self.device)
        a = self.predict(stu_feat, cand_books_index, cand_books_feat)[0]
        ground_book_index = cand_books_index[ground_book_label]
        self.update(ground_book_index, ground_book_label, a == ground_book_label, stu_feat, cand_books_feat)
        return a

    def update(self, ground_book_index, ground_book_label, reward, user, cand_books_feat):
        """
        Updates algorithm's parameters(matrices) : A,b
        Parameters
        ----------
        displayed : index
            displayed article index relative to the pool
        reward : binary
            user clicked or not
        user : array
            user features
        pool_idx : array of indexes
            pool indexes for article identification
        """
        
        x = torch.cat((user, cand_books_feat[ground_book_label])).unsqueeze(-1) # [cand_size, user_dim + book_dim, 1]
        self.A[ground_book_index] += (x @ x.t()).cpu() # [cand_size, user_dim + book_dim, user_dim + book_dim]
        self.b[ground_book_index] += (reward * x.squeeze(-1))
    
    @staticmethod
    def load(file, device='cpu'):
        pt = torch.load(file, map_location=device)
        model = LinUCB(pt['A'].shape[0], pt['feat_size'][0], pt['feat_size'][1], pt['alpha'])
        model.A = pt['A']
        model.b = pt['b']
        return model
    
    def save(self, file):
        torch.save({'A': self.A, 'b':self.b, 'd': self.d, 'alpha': self.alpha, 'feat_size': self.feat_size}, file)