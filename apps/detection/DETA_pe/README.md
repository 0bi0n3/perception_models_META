# SOTA COCO Object Detection with PE

## Getting started

Please refer to [INSTALL.md](../INSTALL.md) for installation and dataset preparation instructions.

Also install [Deformable Attention](models/ops/make.sh) ops.

## Results and Fine-tuned Models

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">detector</th>
<th valign="bottom">vision encoder</th>
<th valign="bottom">box<br/>AP</th>
<th valign="bottom">box(TTA)<br/>AP</th>
<th valign="bottom">download</th>
<!-- TABLE BODY -->
<!-- ROW: DETA -->
 <tr><td align="left">DETA</td>
<td align="center">PE spatial G</td>
<td align="center"> 65.1 </td>
<td align="center"> 65.7 </td>
<td align="center"><a href="https://huggingface.co/facebook/PE-Detection/resolve/main/deta_coco.pth">model</a></td>
</tr>
</tbody></table>


## Training
We apply a three-stage training, Objects365(12ep, 1024pix), Objects365(6ep, 1536pix) and COCO(12ep, 1728pix)

```
sbatch scripts/pretrain_spatial_Gwin384_o365ep12_1024pix_16node.sh

sbatch scripts/pretrain_continue_spatial_Gwin384_o365ep6_1536pix_16node.sh

sbatch scripts/finetune_spatial_Gwin384_o365ep12_1728pix_ep12_8node.sh

```

## Evaluation
```
bash scripts/eval.sh --resume deta_coco.pth
```

## Evaluation with TTA (Test-Time Augmentation)
```
sbatch scripts/eval_tta_slurm.sh --resume deta_coco.pth
```
