import torch
import copy
import numpy as np
import pandas as pd
from torch.utils import data
from torch.utils.data._utils.collate import default_collate



class TCRDataset(data.Dataset):
    """Custom data.Dataset compatible with data.DataLoader. for TCR data."""
    def __init__(self, csv_file, tokenizer, device, target_binder=None, target_peptide=None, excluded_peptide=None, mhctok=None, chain='both'):#, alpha_maxlength, beta_maxlength, epitope_maxlength):
        self.device=device
        self.tokenizer = tokenizer
        df = pd.read_csv(csv_file)
        
        if target_binder:
            df = df[df["binder"]==1]

        if target_peptide:
            df = df[df["peptide"].apply(lambda x: x == target_peptide)]

        if excluded_peptide:
            print("exluded", excluded_peptide)
            iii = df["peptide"].apply(lambda x: x in excluded_peptide)
            df = df[~iii]

        if chain=='alpha': df = df[df["CDR3a"] != '<MIS>']
        elif chain=='beta': df = df[df["CDR3b"] != '<MIS>']
        self.alpha = list(df["CDR3a"])
        self.beta = list(df["CDR3b"])
        self.peptide = list(df["peptide"])
        self.binder = list(df["binder"])

        if mhctok:
            self.mhctok = mhctok
            self.MHC = list(df["MHC"])
        self.df = df
        self.reweight=False
        self.chain_masking_proba=0.0

    @classmethod
    def empty_init(cls, tokenizer, device, mhctok=None ):
        """        Create an empty instance of the class, with no data. """
        obj = cls.__new__(cls)  # Does not call __init__
        super(TCRDataset, obj).__init__()  # Don't forget to call any polymorphic base class initializers
        obj.device=device
        obj.tokenizer = tokenizer
        obj.mhctok = mhctok
        obj.MHC = []
        obj.alpha = []
        obj.beta = []
        obj.peptide = []
        obj.binder = []
        obj.reweight=False
        obj.chain_masking_proba=0.0
        return obj

    def generate_unconditional_data(self, mask_alpha=True, mask_beta=True, mask_peptide=True, mask_mhc=False):
        """Generate a new dataset with the same data, but with some of the data masked. """
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if mask_alpha:
                alpha = '<MIS>'
            else:
                alpha = self.alpha[i]
            if mask_beta:
                beta = '<MIS>'
            else:
                beta = self.beta[i]
            if mask_peptide:
                peptide = '<MIS>'
            else:
                peptide = self.peptide[i]
            if mask_mhc:
                mhc = '<MIS>' 
            else:
                mhc = self.MHC[i]
            new.append(MHC=mhc, alpha=alpha, beta=beta, peptide=peptide, binder=self.binder[i])   

                # new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
        return new
    



    def append(self, MHC='<MIS>', alpha='<MIS>', beta='<MIS>', peptide='<MIS>', binder=0):
        self.MHC.append(MHC)
        self.alpha.append(alpha)
        self.beta.append(beta)
        self.peptide.append(peptide)
        self.binder.append(binder)

    def concatenate(self, tcrdata, inplace = True):
        if inplace: 
            self.MHC += tcrdata.MHC
            self.alpha += tcrdata.alpha
            self.beta += tcrdata.beta
            self.peptide += tcrdata.peptide
            self.binder += tcrdata.binder
        else:
            new = copy.deepcopy(self)
            new.MHC += tcrdata.MHC
            new.alpha += tcrdata.alpha
            new.beta += tcrdata.beta
            new.peptide += tcrdata.peptide
            new.binder += tcrdata.binder
            return new

    def to_pandas(self):
        return pd.DataFrame({"MHC":self.MHC, "CDR3a":self.alpha, "CDR3b":self.beta, "peptide":self.peptide, "binder":self.binder})
    
    def select_binder(self, target_binder=1):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.binder[i] == target_binder:
                new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
        return new
    
    def select_peptide(self, target_peptide ):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.peptide[i] in target_peptide:
                new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
        return new
    
    def select_chain(self, target_chain:str='both'):
        """
        target_chain: 'both', 'alpha', 'beta'

        """
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        if target_chain == 'both':
            for i in range(len(self)):
                if self.alpha[i] == '<MIS>':
                    continue
                if self.beta[i]== '<MIS>':
                    continue
                else:
                    new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
            return new
        if target_chain == 'alpha':
            for i in range(len(self)):
                if self.alpha[i] == '<MIS>':
                    continue
                else:
                    new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
            return new
        if target_chain == 'beta':
            for i in range(len(self)):
                if self.beta[i]== '<MIS>':
                    continue
                else:
                    new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
            return new

    
    def filter_peptide(self, target_peptide):
        new = self.__class__.empty_init(self.tokenizer, self.device, self.mhctok)
        for i in range(len(self)):
            if self.peptide[i] not in target_peptide:
                new.append(MHC=self.MHC[i], alpha=self.alpha[i], beta=self.beta[i], peptide=self.peptide[i], binder=self.binder[i])
        return new
    
    @classmethod
    def from_pandas(cls, df, tokenizer, device, mhctok=None):
        obj = cls.__new__(cls)  # Does not call __init__
        super(TCRDataset, obj).__init__()
        obj.device=device
        obj.tokenizer = tokenizer
        obj.mhctok = mhctok

        obj.alpha = list(df["CDR3a"])
        obj.beta = list(df["CDR3b"])
        obj.peptide = list(df["peptide"])
        obj.binder = list(df["binder"])

        if mhctok:
            obj.mhctok = mhctok
            obj.MHC = list(df["MHC"])
        obj.df = df
        obj.reweight=False
        obj.chain_masking_proba=0.0

        return obj
    
    def set_chain_masking_proba(self, proba=0.0):
        self.chain_masking_proba = proba

    def __getitem__(self, offset):
        """Return one datapoint from the dataset, at position offset in the table.
            - if reweight is True, will provide a weight for each datapoint.
            - if mhctok is provided will provide an mhc token for each datapoint.
        """
        alpha = self.alpha[offset]
        beta = self.beta[offset]
        peptide = self.peptide[offset]
        binder = self.binder[offset]
        if self.chain_masking_proba > 0.0:
            if alpha != '<MIS>' and beta != '<MIS>':
                rd = np.random.uniform()
                if rd < self.chain_masking_proba/2:
                    alpha = '<MIS>'
                elif rd < self.chain_masking_proba:
                    beta = '<MIS>'
            if alpha != '<MIS>' or beta != '<MIS>':
                rd = np.random.uniform()
                if rd < self.chain_masking_proba/2:
                    peptide = '<MIS>'
        if self.mhctok:
            mhc = self.MHC[offset]
            # if self.reweight:
            #     w = self.weights[offset]
            #     return alpha, beta, peptide, binder, mhc, w
            return alpha, beta, peptide, binder, mhc
        return alpha, beta, peptide, binder

    def __len__(self):
        return len(self.peptide)

    def set_reweight(self,alpha):
        """Set the weights for each datapoint, based on the frequency of the peptide in the dataset."""
        freq = self.df["peptide"].value_counts()/self.df["peptide"].value_counts().sum()
        alpha = alpha
        freq = alpha*freq + (1-alpha)/len(self.df["peptide"].value_counts())
        self.weights = (1/torch.tensor(list(self.df.apply(lambda x: freq[x["peptide"]],1 ))))/len(self.df["peptide"].value_counts())
        self.reweight = True

    def all2allmhc_collate_function(self, batch):
        """Collate function for the Tulip model returning peptide, alpha, beta, binder, mhc and weight if reweight is True"""

        if self.reweight:
            (alpha, beta, peptide, binder, mhc, weight) = zip(*batch)
        else:
            (alpha, beta, peptide, binder, mhc) = zip(*batch)

        peptide = self.tokenizer(list(peptide),padding="longest", add_special_tokens=True)
        peptide = {k: torch.tensor(v).to(self.device) for k, v in peptide.items()}#default_collate(peptide)
        
        beta = self.tokenizer(list(beta),  padding="longest", add_special_tokens=True)
        beta = {k: torch.tensor(v).to(self.device) for k, v in beta.items()}

        alpha = self.tokenizer(list(alpha), padding="longest", add_special_tokens=True)
        alpha = {k: torch.tensor(v).to(self.device) for k, v in alpha.items()}
        
        binder =  default_collate(binder).to(self.device)
        mhc = self.mhctok(list(mhc))#default_collate(self.mhctok(list(mhc))['input_ids'])
        mhc = {k: torch.tensor(v).to(self.device) for k, v in mhc.items()}
        # print(mhc)
        if self.reweight:
            weight = torch.tensor(weight).to(self.device)
            return peptide, alpha, beta, binder, mhc, weight

        return peptide, alpha, beta, binder, mhc
