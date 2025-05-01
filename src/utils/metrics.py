import torch
from torchmetrics import Metric
import numpy as np

class DSTMetrics(Metric):
    def __init__(self, slot_temp,  dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.slot_temp = slot_temp
        self.gating_dict = {"none": 0, "ptr": 1, "class": 2}
    
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("turn_acc", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("joint_acc", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("F1_pred", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("F1_count", default=torch.tensor(0), dist_reduce_fx="sum")
    def update(self, all_prediction):
        from_which = "pred_bs_ptr"
        for d, v in all_prediction.items():
            for t in range(len(v)):
                cv = v[t]
                if set(cv["turn_belief"]) == set(cv[from_which]):
                    self.joint_acc += 1
                self.total += 1

                # Compute prediction slot accuracy
                self.temp_acc = self.compute_acc(set(cv["turn_belief"]), set(cv[from_which]), self.slot_temp)
                self.turn_acc += self.temp_acc

                # Compute prediction joint F1 score
                temp_f1, temp_r, temp_p, count = self.compute_prf(set(cv["turn_belief"]), set(cv[from_which]))
                self.F1_pred += temp_f1
                self.F1_count += count
    
    def compute(self):
        joint_acc_score = self.joint_acc / float(self.total) if self.total!=0 else 0
        turn_acc_score = self.turn_acc / float(self.total) if self.total!=0 else 0
        F1_score = self.F1_pred / float(self.F1_count) if self.F1_count!=0 else 0
        return joint_acc_score, F1_score, turn_acc_score

    def compute_acc(self, gold, pred, slot_temp):
        miss_gold = 0
        miss_slot = []
        for g in gold:
            if g not in pred:
                miss_gold += 1
                miss_slot.append(g.rsplit("-", 1)[0])
        wrong_pred = 0
        for p in pred:
            if p not in gold and p.rsplit("-", 1)[0] not in miss_slot:
                wrong_pred += 1
        ACC_TOTAL = len(slot_temp)
        ACC = len(slot_temp) - miss_gold - wrong_pred
        ACC = ACC / float(ACC_TOTAL)
        return ACC

    def compute_prf(self, gold, pred):
        TP, FP, FN = 0, 0, 0
        if len(gold)!= 0:
            count = 1
            for g in gold:
                if g in pred:
                    TP += 1
                else:
                    FN += 1
            for p in pred:
                if p not in gold:
                    FP += 1
            precision = TP / float(TP+FP) if (TP+FP)!=0 else 0
            recall = TP / float(TP+FN) if (TP+FN)!=0 else 0
            F1 = 2 * precision * recall / float(precision + recall) if (precision+recall)!=0 else 0
        else:
            if len(pred)==0:
                precision, recall, F1, count = 1, 1, 1, 1
            else:
                precision, recall, F1, count = 0, 0, 0, 1
        return F1, recall, precision, count
