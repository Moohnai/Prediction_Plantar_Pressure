import nibabel as nib
import  os
from os import path

def SubDirReader(data_path):
    img_L = []
    img_L_a = []
    img_R=[]
    img_R_a = []
    for file in os.listdir(data_path) :
        if file.endswith('.nii') and (file.find('left')>=0) :
            img_L_a.append(os.path.join(data_path, file))
        elif file.endswith('.nii') and (file.find('right')>=0):
            img_R_a.append(os.path.join(data_path, file))
    img_L_a.sort()
    img_R_a.sort()

    for file in range(len(img_R_a)):
        img_L.append(nib.load(img_L_a[file]).get_data().copy())
        img_R.append(nib.load(img_R_a[file]).get_data().copy())

    return img_R, img_L


def DataReader(path):
    P_R = []
    P_L = []
    HalluxValgus= path + '/CAD_WALK_Hallux_Valgus_PreSurgery/HalluxValgus_PreSugery/'
    for folder in os.listdir(HalluxValgus):
        if os.path.isdir(HalluxValgus + folder):
            data_path = os.path.join(HalluxValgus, folder)
            r, l=SubDirReader(data_path)
            P_R.append(r)
            P_L.append(l)
        else:
            continue



    H_R = []
    H_L = []
    HealthyControls= path + '/CAD_WALK_Healthy_Controls_Dataset/HealthyControls/'
    for folder in os.listdir(HealthyControls):
        if os.path.isdir(HealthyControls + folder):
            data_path = os.path.join(HealthyControls, folder)
            r, l = SubDirReader(data_path)
            H_R.append(r)
            H_L.append(l)
        else:
            continue

    return P_R, P_L, H_R, H_L