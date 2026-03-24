import torch

ckpt_path = '/scratch/prj0000000262/Sanwa_forecast/Sanwa_forecast_ft/checkpoints/FT_tiny_Sanwa_forecast_from1_Sw_fc_bs8_lr3e-06_seed42_20260121_152724/best.ckpt'
ckpt = torch.load(ckpt_path, map_location="cpu")

print("ckpt keys:", ckpt.keys())

hp = ckpt.get("hyper_parameters", {})
print("hyper_parameters.task_type:", hp.get("task_type"))
print("hyper_parameters.num_classes:", hp.get("num_classes"))

sd = ckpt.get("state_dict", {})
w = sd.get("model.cls_head.4.weight", None)
b = sd.get("model.cls_head.4.bias", None)
print("cls_head.4.weight:", None if w is None else tuple(w.shape))
print("cls_head.4.bias  :", None if b is None else tuple(b.shape))
