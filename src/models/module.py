from typing import Any, Dict, Tuple, List

import torch
import torch.nn as nn
import torch.nn.functional as F
from lightning import LightningModule
from src.utils.metrics import DSTMetrics

from torchmetrics import SumMetric
import numpy as np

class BaseModel(LightningModule):
    def __init__(self, 
                 model: torch.nn.Module, 
                 optimizer: torch.optim.Optimizer, 
                 scheduler: torch.optim.lr_scheduler,
                 compile: bool = False,
    ) -> None:
        
        super().__init__()
        self.save_hyperparameters(logger=False)

        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loss = SumMetric()
        self.val_metrics = DSTMetrics(self.model.slot_temp)
        self.test_metrics = DSTMetrics(self.model.slot_temp)
        self.all_prediction = {}
    
    def on_train_start(self) -> None:
        self.train_loss.reset()
    def on_validation_epoch_start(self) -> None:
        self.val_metrics.reset()
        self.all_prediction = {}
    def on_test_epoch_start(self) -> None:
        self.test_metrics.reset()
        self.all_prediction = {}
  
    def training_step(self, batch, batch_idx):
        all_point_outputs, gates, words, class_words = self.model.forward(batch)
        loss = self.model.loss(all_point_outputs, gates, words, class_words, batch)
        self.train_loss.update(loss)

        self.log("train/loss", self.train_loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("epoch", self.current_epoch, prog_bar=False, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        _, gates, words, class_words  = self.model.forward(batch)
        self.update_all_prediction(batch, words, gates)
        self.val_metrics.update(self.all_prediction)

    def on_validation_epoch_end(self):
        joint_acc_score, F1_score, turn_acc_score = self.val_metrics.compute()
        self.log("val/jga", joint_acc_score, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/sa", turn_acc_score, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1", F1_score, prog_bar=False, on_epoch=True, on_step=False)

    def test_step(self, batch, batch_idx):
        _, gates, words, class_words = self.model.forward(batch)
        self.update_all_prediction(batch, words, gates)
        self.test_metrics.update(self.all_prediction)
    
    def on_test_epoch_end(self):
        joint_acc_score, F1_score, turn_acc_score = self.test_metrics.compute()
        self.log("val/jga", joint_acc_score, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/sa", turn_acc_score, prog_bar=True, on_epoch=True, on_step=False)
        self.log("val/f1", F1_score, prog_bar=False, on_epoch=True, on_step=False)

    def update_all_prediction(self, batch, words, gates):
        for bi in range(len(batch["context_len"])):
            if batch["ID"][bi] not in self.all_prediction.keys():
                self.all_prediction[batch["ID"][bi]] = {}
            self.all_prediction[batch["ID"][bi]][batch["turn_id"][bi]] = {"turn_belief":batch["turn_belief"][bi]}
            predict_belief_bsz_ptr, predict_belief_bsz_class = [], []
            gate = torch.argmax(gates.transpose(0, 1)[bi], dim=1)

            # pointer-generator results
            if self.model.use_gate:
                for si, sg in enumerate(gate):
                    if sg==self.model.gating_dict["none"]:
                        continue
                    elif sg==self.model.gating_dict["ptr"]:
                        pred = np.transpose(words[si])[bi]
                        st = []
                        for e in pred:
                            if e== 'EOS': break
                            else: st.append(e)
                        st = " ".join(st)
                        if st == "none":
                            continue
                        else:
                            predict_belief_bsz_ptr.append(self.model.slot_temp[si]+"-"+str(st))
                    else:
                        predict_belief_bsz_ptr.append(self.model.slot_temp[si]+"-"+self.model.inverse_unpoint_slot[sg.item()])
            else:
                for si, _ in enumerate(gate):
                    pred = np.transpose(words[si])[bi]
                    st = []
                    for e in pred:
                        if e == 'EOS': break
                        else: st.append(e)
                    st = " ".join(st)
                    if st == "none":
                        continue
                    else:
                        predict_belief_bsz_ptr.append(self.model.slot_temp[si]+"-"+str(st))

            self.all_prediction[batch["ID"][bi]][batch["turn_id"][bi]]["pred_bs_ptr"] = predict_belief_bsz_ptr
    
    def setup(self, stage: str) -> None:
        if self.hparams.compile and stage == "fit":
            self.model = torch.compile(self.model)

    def configure_optimizers(self) -> Dict[str, Any]:
        optimizer = self.hparams.optimizer(params=self.trainer.model.parameters())
        if self.hparams.scheduler is not None:
            scheduler = self.hparams.scheduler(optimizer=optimizer)
            return {
                "optimizer": optimizer,
                "lr_scheduler": {
                    "scheduler": scheduler,
                    "monitor": "val/jga",
                    "interval": "epoch",
                    "frequency": 1,
                },
            }
        return {"optimizer": optimizer}

