# -*- coding: utf-8 -*-
from lightning import LightningDataModule
from torch.utils.data import DataLoader
from src.data.components.dataset import collate_fn, ImbalancedDatasetSampler, prepare_data_seq

class DataModule(LightningDataModule):
    def __init__(
        self,
        dataset: str,
        train_batch_size: int,
        eval_batch_size: int,
        num_workers: int,
        all_vocab: int,
        data_ratio: int,
        only_domain: str,
        except_domain: str,
        fisher_sample: int,
        save_dir: str, 
        imbalance_sampler: bool,
    ):
        super().__init__()
        self.dataset = dataset
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_workers = num_workers
        self.all_vocab = all_vocab
        self.data_ratio = data_ratio
        self.only_domain = only_domain
        self.except_domain = except_domain
        self.fisher_sample = fisher_sample
        self.save_dir = save_dir
        self.imbalance_sampler = imbalance_sampler
        self.vocab_size = None
        self.num_slots = None
        self.slot_value_sizes = None

    def setup(self, stage=None):
        train, val, test, test_4d, LANG, SLOTS_LIST, gating_dict, nb_train_vocab = prepare_data_seq(True, 
                                                                                                    self.train_batch_size, 
                                                                                                    self.eval_batch_size, 
                                                                                                    self.save_dir, 
                                                                                                    self.dataset,
                                                                                                    self.all_vocab,
                                                                                                    self.except_domain, 
                                                                                                    self.only_domain,
                                                                                                    self.data_ratio,
                                                                                                    self.fisher_sample,
                                                                                                    
                                                                                                    )
        self.train = train
        self.val = val
        self.test = test
        self.test_4d = test_4d
        self.LANG = LANG
        self.SLOTS_LIST = SLOTS_LIST
        self.gating_dict = gating_dict  
        self.nb_train_vocab = nb_train_vocab
        self.vocab_size = len(LANG[0].word2index)
        self.num_slots = len(SLOTS_LIST)

    def _make_loader(self, dataset, shuffle_flag, batch_size=None):
        if self.imbalance_sampler and shuffle_flag:
            sampler = ImbalancedDatasetSampler(dataset)
            return DataLoader(dataset,
                              batch_size=batch_size,
                              sampler=sampler,
                              collate_fn=collate_fn,
                              num_workers=self.num_workers,
                              pin_memory=True)
        else:
            return DataLoader(dataset,
                              batch_size=batch_size,
                              shuffle=shuffle_flag,
                              collate_fn=collate_fn,
                              num_workers=self.num_workers,
                              pin_memory=True)
        

    def train_dataloader(self):
        return self._make_loader(self.train, shuffle_flag=True, batch_size=self.train_batch_size)

    def val_dataloader(self):
        return self._make_loader(self.val, shuffle_flag=False, batch_size=self.eval_batch_size)

    def test_dataloader(self):
        return self._make_loader(self.test, shuffle_flag=False, batch_size=self.eval_batch_size)

if __name__ == "__main__":
    pass
