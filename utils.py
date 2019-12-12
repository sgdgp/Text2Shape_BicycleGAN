import nrrd
import numpy as np
import uuid
import sys
from PIL import Image
import os

def get_voxel_from_file(nrrd_filename):
    nrrd_tensor, options = nrrd.read(nrrd_filename)
    # print(nrrd_filename)
    # print("Inside get_voxel_from_file")
    # print(options)
    # print(nrrd_tensor)
    voxel_tensor = nrrd_tensor.astype(np.float32) 
    # print(voxel_tensor)
    
    voxel_tensor /= 255.

    # print(voxel_tensor)
    # import sys
    # sys.exit()

    voxel_tensor = np.rollaxis(voxel_tensor, 0, 4)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 1)
    voxel_tensor = np.swapaxes(voxel_tensor, 0, 2)
    # print(type(voxel_tensor))
    return voxel_tensor


def write_temp_voxel_png(voxel,output_folder, voxel_renderer_path):
    temp = str(uuid.uuid4())
    unique_filename = output_folder + temp
    unique_filename_nrrd = output_folder + temp +".nrrd"

    voxel = (voxel * 255).astype(np.uint8)
    voxel = np.clip(voxel, 0, 255)
    nrrd.write(unique_filename_nrrd, voxel)

    command = 'xvfb-run -s "-ac -screen 0 1280x1024x24" '+ voxel_renderer_path +' --input '+unique_filename_nrrd+' --output '+unique_filename
    print(command)
    os.system(command)

    return unique_filename + "_0.png"

# def write_text_to_png(text, id2word,config, output_folder):

def merge_and_write_to_output_folder(f_list,output_filename):
    images = [Image.open(x) for x in f_list]
    widths, heights = zip(*(i.size for i in images))

    total_width = sum(widths)
    max_height = max(heights)

    new_im = Image.new('RGB', (total_width, max_height))

    x_offset = 0
    for im in images:
        new_im.paste(im, (x_offset,0))
        x_offset += im.size[0]

    new_im.save(output_filename)

def remove_temp_files(f_list):
    for f in f_list:
        os.remove(f)


def get_text_from_ids(id_list, d, flag):
    l = []
    # print(id_list)
    for id in id_list:
        if flag :
            l.append(d[int(id)])
        else :
            l.append(d[int (np.argmax(id))])
    
    return str(" ".join(l))






