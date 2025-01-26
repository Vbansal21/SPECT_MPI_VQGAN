from PIL import Image
import os
from pathlib import Path
import torch
import torchvision.transforms as transforms

convert_tensor = transforms.PILToTensor()

def print_all_files(p):
    files = os.listdir(p)
    for i in files:
        print(i)
    print("\n")

def create_scan_folders(p):
    scan_types = ["SA","HLA","VLA"]
    for i in scan_types:
        T = [p,i]
        #print(T)
        new_path = "/".join(T)
        #print(new_path)
        try:
            os.mkdir(new_path)
        except FileExistsError:
            print(T,"exists")
        #print_all_files(new_path)

def create_folders(L,p):
    for i in L:
        i = i[:-4:]
        T = [p,i]
        new_path = "/".join(T)
        try:
            os.mkdir(new_path)
            print("Created",i,"folder")
        except FileExistsError:
            print(i,"exists")


def PathFinder(parentdir, imgname):
    imgname = imgname[:-4:]
    filename = imgname+".dat"
    textfilepath = [parentdir, filename]
    textfilepath = "/".join(textfilepath)
    imgname = imgname+".jpg"
    imagefilepath = [parentdir, imgname]
    imagefilepath = "/".join(imagefilepath)
    return textfilepath

def WriteToFile(filepath, t, Tensor):
    F = open(filepath, mode = "a")
    s = t + "=" + str(Tensor)
    F.write(s)
    F.close()


def convImgToTensor(imgpath):
    path_ = imgpath
    img = Image.open(imgpath)
    #imgpath = []
    parentdir = os.path.dirname(imgpath)
    i = os.path.basename(imgpath)
    imgname = i

    list_of_tensors = [[],[],[],[],[],[],[],[]]

    ## SA (Apex to Base)
    rest = 0
    stress = 0
    for i in range(40):
        if((0 <= i and i < 10) or (20 <= i and i < 30)):
            if( i < 10 ):
                imgpath = [parentdir, imgname, "SA"]
                t = i%10
                x = ((70 + 90*t), 52)
                box = (x[0],x[1],x[0]+88,x[1]+88)
                img1 = img.crop(box)
                t = "stress_row1_"+str(t)
                imgten = convert_tensor(img1)
                list_of_tensors[0].append(imgten)
                stress += 1
            else:
                imgpath = [parentdir, imgname, "SA"]
                t = i%10
                x = ((70 + 90*t), 232)
                box = (x[0],x[1],x[0]+88,x[1]+88)
                img1 = img.crop(box)
                t += 10
                t = "stress_row2_"+str(t)
                imgten = convert_tensor(img1)
                list_of_tensors[1].append(imgten)
                stress += 1
        else:
            if( i < 20):
                imgpath = [parentdir, imgname, "SA"]
                t = i%10
                x = ((70 + 90*t), 142)
                box = (x[0],x[1],x[0]+88,x[1]+88)
                img1 = img.crop(box)
                t = "rest_row1_"+str(t)
                imgten = convert_tensor(img1)
                list_of_tensors[2].append(imgten)
                rest += 1
            else:
                imgpath = [parentdir, imgname, "SA"]
                t = i%10
                x = ((70 + 90*t), 322)
                box = (x[0],x[1],x[0]+88,x[1]+88)
                img1 = img.crop(box)
                t = "rest_row2_"+str(t)
                imgten = convert_tensor(img1)
                list_of_tensors[3].append(imgten)
                rest += 1

    ## HLA (INF to ANT)
    rest = 0
    stress = 0
    for i in range(20):
        if(0 <= i and i < 10):
            imgpath = [parentdir, imgname, "HLA"]
            t = i%10
            x = ((70 + 90*t), 484)
            box = (x[0],x[1],x[0]+88,x[1]+88)
            img1 = img.crop(box)
            t = "stress_row1_"+str(t)
            imgten = convert_tensor(img1)
            list_of_tensors[4].append(imgten)
            stress += 1
        else:
            imgpath = [parentdir, imgname, "HLA"]
            t = i%10
            x = ((70 + 90*t), 574)
            box = (x[0],x[1],x[0]+88,x[1]+88)
            img1 = img.crop(box)
            t = "rest_row1_"+str(t)
            imgten = convert_tensor(img1)
            list_of_tensors[5].append(imgten)
            rest += 1

    ## VLA (SEP to LAT)
    rest = 0
    stress = 0
    for i in range(20):
        if(0 <= i and i < 10):
            imgpath = [parentdir, imgname, "VLA"]
            t = i%10
            x = ((70 + 90*t), 708)
            box = (x[0],x[1],x[0]+88,x[1]+88)
            img1 = img.crop(box)
            t = "stress_row1_"+str(t)
            imgten = convert_tensor(img1)
            list_of_tensors[6].append(imgten)
            stress += 1
        else:
            imgpath = [parentdir, imgname, "VLA"]
            t = i%10
            x = ((70 + 90*t), 798)
            box = (x[0],x[1],x[0]+88,x[1]+88)
            img1 = img.crop(box)
            t = "rest_row1_"+str(t)
            imgten = convert_tensor(img1)
            list_of_tensors[7].append(imgten)
            rest += 1
    l1 = []
    for i in list_of_tensors:
        l1.append(torch.stack([k for k in i],dim=1))
    ten = torch.concat(l1,dim=1).contiguous()
    datatype = None
    if "train" in path_.lower():
        datatype = "train"
    elif "test" in path_.lower():
        datatype = "test"
    elif "val" in path_.lower():
        datatype = "val"
    else: return
    class_name = Path(path_).parts[-2].lower()
    file_name = (Path(path_).parts[-1])[:-4]+".dat"
    path_ = os.path.join(os.curdir,"dataset",datatype,class_name,"")
    file_path = os.path.join(path_,file_name)
    if not os.path.exists(path_):
        print("Creating dir:",path_)
        os.makedirs(path_,exist_ok=True)
    print(path_,file_path,ten.shape)
    torch.save(ten,file_path)


path_ = os.path.join(os.curdir,"archive","SPECT_MPI_Dataset","")
dirs = os.listdir(path_)
dirs = [
    os.path.join(path_,i)
    for i in dirs
    if os.path.isdir(os.path.join(path_,i))]
dirs = [[os.path.join(dir_,file_) for file_ in os.listdir(dir_)] for dir_ in dirs]
tmp = []
for i in dirs:
    for j in i:
        tmp.append(j)
dirs = tmp
dirs = [[os.path.join(dir_,file_) for file_ in os.listdir(dir_)] for dir_ in dirs]
tmp = []
for i in dirs:
    for j in i:
        tmp.append(j)
dirs = tmp

for i in dirs:
    convImgToTensor(i)
