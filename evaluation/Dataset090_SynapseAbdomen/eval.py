import glob
import os
import SimpleITK as sitk
import numpy as np
import argparse
from medpy import metric
import wandb
from nnunetv2.paths import nnUNet_raw
from tqdm import tqdm

DATASET = "Dataset090_SynapseAbdomen"

def read_nii(path):
    return sitk.GetArrayFromImage(sitk.ReadImage(path))

def dice(pred, label):
    if (pred.sum() + label.sum()) == 0:
        return 100
    else:
        return 200. * np.logical_and(pred, label).sum() / (pred.sum() + label.sum())
def hd(pred,gt):
        if pred.sum() > 0 and gt.sum()>0:
            hd95 = metric.binary.hd95(pred, gt)
            return  hd95
        else:
            return 0
            
def process_label(label):
    spleen = label == 1
    right_kidney = label == 2
    left_kidney = label == 3
    gallbladder = label == 4
    liver = label == 6
    stomach = label == 7
    aorta = label == 8
    pancreas = label == 11
   
    return spleen,right_kidney,left_kidney,gallbladder,liver,stomach,aorta,pancreas

def test(model, fold):
    if model == "":
        infer_path = f'inferTs'
    else:
        infer_path = f'inferTs_{model}_{fold}'

    label_list = sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, 'labelsTs', '*nii.gz')))
    infer_list = sorted(glob.glob(os.path.join(nnUNet_raw, DATASET, infer_path, '*nii.gz')))
    print(os.path.join(nnUNet_raw, DATASET, infer_path, '*nii.gz'))
    print(f'Found {len(label_list)} labels and {len(infer_list)} gts')
    print("loading success...")

    Dice_spleen=[]
    Dice_right_kidney=[]
    Dice_left_kidney=[]
    Dice_gallbladder=[]
    Dice_liver=[]
    Dice_stomach=[]
    Dice_aorta=[]
    Dice_pancreas=[]
    
    hd_spleen=[]
    hd_right_kidney=[]
    hd_left_kidney=[]
    hd_gallbladder=[]
    hd_liver=[]
    hd_stomach=[]
    hd_aorta=[]
    hd_pancreas=[]

    file=os.path.join(nnUNet_raw, DATASET, infer_path)
    fw = open(file+'/dice_pre.txt', 'a')
    
    for label_path,infer_path in tqdm(zip(label_list,infer_list), total=len(label_list)):
        #print(label_path.split('/')[-1])
        #print(infer_path.split('/')[-1])
        label,infer = read_nii(label_path),read_nii(infer_path)
        label_spleen,label_right_kidney,label_left_kidney,label_gallbladder,label_liver,label_stomach,label_aorta,label_pancreas=process_label(label)
        infer_spleen,infer_right_kidney,infer_left_kidney,infer_gallbladder,infer_liver,infer_stomach,infer_aorta,infer_pancreas=process_label(infer)
        
        Dice_spleen.append(dice(infer_spleen,label_spleen))
        Dice_right_kidney.append(dice(infer_right_kidney,label_right_kidney))
        Dice_left_kidney.append(dice(infer_left_kidney,label_left_kidney))
        Dice_gallbladder.append(dice(infer_gallbladder,label_gallbladder))
        Dice_liver.append(dice(infer_liver,label_liver))
        Dice_stomach.append(dice(infer_stomach,label_stomach))
        Dice_aorta.append(dice(infer_aorta,label_aorta))
        Dice_pancreas.append(dice(infer_pancreas,label_pancreas))
        
        hd_spleen.append(hd(infer_spleen,label_spleen))
        hd_right_kidney.append(hd(infer_right_kidney,label_right_kidney))
        hd_left_kidney.append(hd(infer_left_kidney,label_left_kidney))
        hd_gallbladder.append(hd(infer_gallbladder,label_gallbladder))
        hd_liver.append(hd(infer_liver,label_liver))
        hd_stomach.append(hd(infer_stomach,label_stomach))
        hd_aorta.append(hd(infer_aorta,label_aorta))
        hd_pancreas.append(hd(infer_pancreas,label_pancreas))
        
        # fw.write('*'*20+'\n',)
        # fw.write(infer_path.split('/')[-1]+'\n')
        # fw.write('Dice_spleen: {:.4f}\n'.format(Dice_spleen[-1]))
        # fw.write('Dice_right_kidney: {:.4f}\n'.format(Dice_right_kidney[-1]))
        # fw.write('Dice_left_kidney: {:.4f}\n'.format(Dice_left_kidney[-1]))
        # fw.write('Dice_gallbladder: {:.4f}\n'.format(Dice_gallbladder[-1]))
        # fw.write('Dice_liver: {:.4f}\n'.format(Dice_liver[-1]))
        # fw.write('Dice_stomach: {:.4f}\n'.format(Dice_stomach[-1]))
        # fw.write('Dice_aorta: {:.4f}\n'.format(Dice_aorta[-1]))
        # fw.write('Dice_pancreas: {:.4f}\n'.format(Dice_pancreas[-1]))
        # 
        # fw.write('hd_spleen: {:.4f}\n'.format(hd_spleen[-1]))
        # fw.write('hd_right_kidney: {:.4f}\n'.format(hd_right_kidney[-1]))
        # fw.write('hd_left_kidney: {:.4f}\n'.format(hd_left_kidney[-1]))
        # fw.write('hd_gallbladder: {:.4f}\n'.format(hd_gallbladder[-1]))
        # fw.write('hd_liver: {:.4f}\n'.format(hd_liver[-1]))
        # fw.write('hd_stomach: {:.4f}\n'.format(hd_stomach[-1]))
        # fw.write('hd_aorta: {:.4f}\n'.format(hd_aorta[-1]))
        # fw.write('hd_pancreas: {:.4f}\n'.format(hd_pancreas[-1]))
        
        dsc=[]
        HD=[]
        dsc.append(Dice_spleen[-1])
        dsc.append((Dice_right_kidney[-1]))
        dsc.append(Dice_left_kidney[-1])
        dsc.append(np.mean(Dice_gallbladder[-1]))
        dsc.append(np.mean(Dice_liver[-1]))
        dsc.append(np.mean(Dice_stomach[-1]))
        dsc.append(np.mean(Dice_aorta[-1]))
        dsc.append(np.mean(Dice_pancreas[-1]))
        # fw.write('DSC:'+str(np.mean(dsc))+'\n')
        
        HD.append(hd_spleen[-1])
        HD.append(hd_right_kidney[-1])
        HD.append(hd_left_kidney[-1])
        HD.append(hd_gallbladder[-1])
        HD.append(hd_liver[-1])
        HD.append(hd_stomach[-1])
        HD.append(hd_aorta[-1])
        HD.append(hd_pancreas[-1])
        # fw.write('hd:'+str(np.mean(HD))+'\n')
        
    
    fw.write('*'*20+'\n')
    fw.write('Mean_Dice\n')
    fw.write('Dice_aorta'+str(np.mean(Dice_aorta))+'\n')
    fw.write('Dice_gallbladder'+str(np.mean(Dice_gallbladder))+'\n')
    fw.write('Dice_left_kidney'+str(np.mean(Dice_left_kidney))+'\n')
    fw.write('Dice_right_kidney'+str(np.mean(Dice_right_kidney))+'\n')
    fw.write('Dice_liver'+str(np.mean(Dice_liver))+'\n')
    fw.write('Dice_pancreas'+str(np.mean(Dice_pancreas))+'\n')
    fw.write('Dice_spleen'+str(np.mean(Dice_spleen))+'\n')
    fw.write('Dice_stomach'+str(np.mean(Dice_stomach))+'\n')
    
    fw.write('Mean_hd\n')
    fw.write('hd_aorta'+str(np.mean(hd_aorta))+'\n')
    fw.write('hd_gallbladder'+str(np.mean(hd_gallbladder))+'\n')
    fw.write('hd_left_kidney'+str(np.mean(hd_left_kidney))+'\n')
    fw.write('hd_right_kidney'+str(np.mean(hd_right_kidney))+'\n')
    fw.write('hd_liver'+str(np.mean(hd_liver))+'\n')
    fw.write('hd_pancreas'+str(np.mean(hd_pancreas))+'\n')
    fw.write('hd_spleen'+str(np.mean(hd_spleen))+'\n')
    fw.write('hd_stomach'+str(np.mean(hd_stomach))+'\n')
   
    fw.write('*'*20+'\n')
    
    dsc=[]
    dsc.append(np.mean(Dice_spleen))
    dsc.append(np.mean(Dice_right_kidney))
    dsc.append(np.mean(Dice_left_kidney))
    dsc.append(np.mean(Dice_gallbladder))
    dsc.append(np.mean(Dice_liver))
    dsc.append(np.mean(Dice_stomach))
    dsc.append(np.mean(Dice_aorta))
    dsc.append(np.mean(Dice_pancreas))
    fw.write('Average DICE:'+str(np.mean(dsc))+'\n')
    
    HD=[]
    HD.append(np.mean(hd_spleen))
    HD.append(np.mean(hd_right_kidney))
    HD.append(np.mean(hd_left_kidney))
    HD.append(np.mean(hd_gallbladder))
    HD.append(np.mean(hd_liver))
    HD.append(np.mean(hd_stomach))
    HD.append(np.mean(hd_aorta))
    HD.append(np.mean(hd_pancreas))
    fw.write('Average HD:'+str(np.mean(HD))+'\n')

    wandb.log({
        'Test/Average_Dice': np.mean(dsc),
        'Test/Average_HD95': np.mean(HD),
        'Test/Dice_Spleen': dsc[0],
        'Test/Dice_RightKidney': dsc[1],
        'Test/Dice_LeftKidney': dsc[2],
        'Test/Dice_Gallbladder': dsc[3],
        'Test/Dice_Liver': dsc[4],
        'Test/Dice_Stomach': dsc[5],
        'Test/Dice_Aorta': dsc[6],
        'Test/Dice_Pancreas': dsc[7],
        'Test/HD95_Spleen': HD[0],
        'Test/HD95_RightKidney': HD[1],
        'Test/HD95_LeftKidney': HD[2],
        'Test/HD95_Gallbladder': HD[3],
        'Test/HD95_Liver': HD[4],
        'Test/HD95_Stomach': HD[5],
        'Test/HD95_Aorta': HD[6],
        'Test/HD95_Pancreas': HD[7],
        'Test/Epoch': 299,
    })
    
    print('done')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("model", help="model name")
    parser.add_argument("fold", help="fold name")
    parser.add_argument("config", help="config")
    args = parser.parse_args()

    name_and_id = f'Dataset090_SynapseAbdomen_nnUNetTrainer{args.model}_{args.config}_Fold{args.fold}'
    run = wandb.init(
        #project="test",
        #name=name_and_id,
        #entity="test",
        #id=name_and_id,
        #resume="allow",
        mode="disabled"
    )

    test(args.model, args.fold)
