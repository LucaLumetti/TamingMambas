import glob
import os
import SimpleITK as sitk
import numpy as np
from medpy.metric import binary
import argparse
import wandb
from tqdm import tqdm
from nnunetv2.paths import nnUNet_raw

DATASET = "Dataset027_ACDC"

def read_nii(path):
    itk_img=sitk.ReadImage(path)
    spacing=np.array(itk_img.GetSpacing())
    return sitk.GetArrayFromImage(itk_img),spacing

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 1
    else:
        return 2. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())

def process_label(label):
    rv = label == 1
    myo = label == 2
    lv = label == 3
    
    return rv, myo, lv

def hd(pred,gt):
    if pred.sum() > 0 and gt.sum()>0:
        hd95 = binary.hd95(pred, gt)
        return  hd95
    else:
        return 0

def test(model, fold):
    if model == "":
        infer_path = f'inferTs'
    else:
        infer_path = f'inferTs_{model}_{fold}'

    label_list=sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, 'labelsTs', '*nii.gz')))
    infer_list=sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, infer_path, '*nii.gz')))

    print(os.path.join(nnUNet_raw, DATASET, infer_path, '*nii.gz'))

    print(f'Found {len(label_list)} labels and {len(infer_list)} gts')


    Dice_rv=[]
    Dice_myo=[]
    Dice_lv=[]
    
    hd_rv=[]
    hd_myo=[]
    hd_lv=[]

    file = os.path.join(nnUNet_raw, DATASET, infer_path)
    fw = open(file + '/dice_pre.txt', 'a')
    
    for label_path,infer_path in tqdm(zip(label_list,infer_list)):
        label,spacing= read_nii(label_path)
        infer,spacing= read_nii(infer_path)
        label_rv,label_myo,label_lv=process_label(label)
        infer_rv,infer_myo,infer_lv=process_label(infer)
        
        Dice_rv.append(dice(infer_rv,label_rv))
        Dice_myo.append(dice(infer_myo,label_myo))
        Dice_lv.append(dice(infer_lv,label_lv))
        
        hd_rv.append(hd(infer_rv,label_rv))
        hd_myo.append(hd(infer_myo,label_myo))
        hd_lv.append(hd(infer_lv,label_lv))
    
    dsc=[]
    dsc.append(np.mean(Dice_rv))
    dsc.append(np.mean(Dice_myo))
    dsc.append(np.mean(Dice_lv))
    avg_hd=[]
    avg_hd.append(np.mean(hd_rv))
    avg_hd.append(np.mean(hd_myo))
    avg_hd.append(np.mean(hd_lv))

    fw.write('*' * 20 + '\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_RV' + str(dsc[0]) + '\n')
    fw.write('Dice_MYO' + str(dsc[1]) + '\n')
    fw.write('Dice_LV' + str(dsc[2]) + '\n')

    fw.write('Mean_hd\n')
    fw.write('hd_RV' + str(np.mean(avg_hd[0])) + '\n')
    fw.write('hd_MYO' + str(np.mean(avg_hd[1])) + '\n')
    fw.write('hd_LV' + str(np.mean(avg_hd[2])) + '\n')

    fw.write('*' * 20 + '\n')

    fw.write('Average DICE:' + str(np.mean(dsc)) + '\n')

    fw.write('Average HD:' + str(np.mean(avg_hd)) + '\n')

    wandb.log({
        'Test/Average_Dice': np.mean(dsc),
        'Test/Average_HD95': np.mean(avg_hd),
        'Test/Dice_RV': dsc[0],
        'Test/Dice_MYO': dsc[1],
        'Test/Dice_LV': dsc[2],
        'Test/HD95_RV': avg_hd[0],
        'Test/HD95_MYO': avg_hd[1],
        'Test/HD95_LV': avg_hd[2],
        'Test/Epoch': 999,
    })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("fold", help="fold name")
    parser.add_argument("config", help="config")
    args = parser.parse_args()


    name_and_id = f'Dataset027_ACDC_nnUNetTrainer{args.model}_{args.config}_Fold{args.fold}'
    run = wandb.init(
        #project="test",
        #name=name_and_id,
        #entity="test",
        #id=name_and_id,
        #resume="allow",
        mode="disabled"
    )

    test(args.model, args.fold)
