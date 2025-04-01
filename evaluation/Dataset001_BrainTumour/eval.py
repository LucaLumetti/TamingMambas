import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy.metric import binary
import wandb
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm

DATASET = "Dataset001_BrainTumour"

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def new_dice(pred,label):
    tp_hard = np.sum((pred == 1).astype(np.float) * (label == 1).astype(np.float))
    fp_hard = np.sum((pred == 1).astype(np.float) * (label != 1).astype(np.float))
    fn_hard = np.sum((pred != 1).astype(np.float) * (label == 1).astype(np.float))
    return 2*tp_hard/(2*tp_hard+fp_hard+fn_hard)
    
def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
        
def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0
        
def process_label(label):
    net = label == 2
    ed = label == 1
    et = label == 3
    ET=et
    TC=net+et
    WT=net+et+ed
    return ET,TC,WT

def test(model, fold):
    if model == "":
        infer_path = f'inferTs'
    else:
        infer_path = f'inferTs_{model}_{fold}'

    label_list=sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, 'labelsTs', '*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, infer_path, '*nii.gz')))
    print(os.path.join(nnUNet_raw, DATASET, infer_path))
    print(f'Found {len(label_list)} labels and {len(infer_list)} gts')
    print("loading success...")
    Dice_et=[]
    Dice_tc=[]
    Dice_wt=[]
    
    HD_et=[]
    HD_tc=[]
    HD_wt=[]
    file=os.path.join(nnUNet_raw, DATASET, infer_path)
    fw = open(file+'/dice_pre.txt', 'a')

    for label_path,infer_path in tqdm(zip(label_list,infer_list)):
        #print(label_path.split('/')[-1])
        #print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_et,label_tc,label_wt=process_label(label)
        infer_et,infer_tc,infer_wt=process_label(infer)
        Dice_et.append(dice(infer_et,label_et))
        Dice_tc.append(dice(infer_tc,label_tc))
        Dice_wt.append(dice(infer_wt,label_wt))
        
        HD_et.append(hd(infer_et,label_et))
        HD_tc.append(hd(infer_tc,label_tc))
        HD_wt.append(hd(infer_wt,label_wt))
        
        fw.write('*'*20+'\n',)
        fw.write(infer_path.split('/')[-1]+'\n')
        fw.write('hd_et: {:.4f}\n'.format(HD_et[-1]))
        fw.write('hd_tc: {:.4f}\n'.format(HD_tc[-1]))
        fw.write('hd_wt: {:.4f}\n'.format(HD_wt[-1]))
        fw.write('*'*20+'\n',)
        fw.write('Dice_et: {:.4f}\n'.format(Dice_et[-1]))
        fw.write('Dice_tc: {:.4f}\n'.format(Dice_tc[-1]))
        fw.write('Dice_wt: {:.4f}\n'.format(Dice_wt[-1]))

        #print('dice_et: {:.4f}'.format(np.mean(Dice_et)))
        #print('dice_tc: {:.4f}'.format(np.mean(Dice_tc)))
        #print('dice_wt: {:.4f}'.format(np.mean(Dice_wt)))
    dsc=[]
    avg_hd=[]
    dsc.append(np.mean(Dice_et))
    dsc.append(np.mean(Dice_tc))
    dsc.append(np.mean(Dice_wt))
    
    avg_hd.append(np.mean(HD_et))
    avg_hd.append(np.mean(HD_tc))
    avg_hd.append(np.mean(HD_wt))

    wandb.log({
        'Test/Average_Dice': np.mean(dsc),
        'Test/Average_HD95': np.mean(avg_hd),
        'Test/Dice_ET': dsc[0],
        'Test/Dice_TC': dsc[1],
        'Test/Dice_WT': dsc[2],
        'Test/HD95_ET': avg_hd[0],
        'Test/HD95_TC': avg_hd[1],
        'Test/HD95_WT': avg_hd[2],
        'Test/Epoch': 999
    })
    
    fw.write('Dice_et'+str(np.mean(Dice_et))+' '+'\n')
    fw.write('Dice_tc'+str(np.mean(Dice_tc))+' '+'\n')
    fw.write('Dice_wt'+str(np.mean(Dice_wt))+' '+'\n')
    
    fw.write('HD_et'+str(np.mean(HD_et))+' '+'\n')
    fw.write('HD_tc'+str(np.mean(HD_tc))+' '+'\n')
    fw.write('HD_wt'+str(np.mean(HD_wt))+' '+'\n')
    
    fw.write('Dice'+str(np.mean(dsc))+' '+'\n')
    fw.write('HD'+str(np.mean(avg_hd))+' '+'\n')
    #print('Dice'+str(np.mean(dsc))+' '+'\n')
    #print('HD'+str(np.mean(avg_hd))+' '+'\n')
    


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("fold", help="fold name")
    parser.add_argument("config", help="config")
    args = parser.parse_args()

    name_and_id = f'Dataset001_BrainTumour_nnUNetTrainer{args.model}_{args.config}_Fold{args.fold}'
    run = wandb.init(
        project="MambaWACV",
        name=name_and_id,
        entity="maxillo",
        id=name_and_id,
        resume="allow",
    )

    test(args.model, args.fold)
