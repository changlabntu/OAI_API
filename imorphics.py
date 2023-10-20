import numpy as np
from utils.images_utils import imagesc
import numpy as np
from matplotlib.path import Path
import tifffile as tiff
import os, glob


def get_data(lines, index=0):
    data = []
    while index < len(lines):
        line = lines[index].strip()
        if line.startswith('{'):
            block_data, index = get_data(lines, index + 1)
            data.append(block_data)
        elif line.startswith('}'):
            return data, index
        elif line:  # non-empty line
            try:
                # try to convert the line to a list of floats (a point)
                point = list(map(float, line.split()))
                if data and isinstance(data[-1], np.ndarray):
                    # if the last item in data is a numpy array, stack the point to it
                    data[-1] = np.vstack([data[-1], point])
                else:
                    # otherwise, this is the first point in a new block, create a new numpy array
                    data.append(np.array(point).reshape(1, -1))
            except ValueError:
                # line could not be converted to a list of floats, ignore it
                pass
        index += 1
    return data, index


def structured_text_to_dict(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()
    data, _ = get_data(lines)
    return data


def load_imorphics_mask(txt_path, ratio=1):
    nx, ny = 384 * ratio, 384 * ratio
    txt = structured_text_to_dict(txt_path)
    masks = np.zeros((160, nx, ny), dtype=np.int64)

    for c in range(len(txt)):
        for z_dumb in range(len(txt[c])):
            for i in range(len(txt[c][z_dumb])):
                print((c, z_dumb, i))
                a_piece = txt[c][z_dumb][i][0]

                poly_verts = [(x, y) for x, y in zip(a_piece[:,0] * ratio, ny - ratio * (a_piece[:,1] + 1))]
                assert a_piece[:,2].var() == 0
                z = int(a_piece[:,2].mean())

                # Create vertex coordinates for each grid cell...
                # (<0,0> is at the top left of the grid in this system)
                x, y = np.meshgrid(np.arange(nx), np.arange(ny))
                x, y = x.flatten(), y.flatten()

                points = np.vstack((x,y)).T

                path = Path(poly_verts)
                grid = path.contains_points(points)
                grid = grid.reshape((ny,nx))
                masks[z, :, :] += grid * (c + 1)
    return masks


root = '/media/ghc/GHc_data2/imorphics/'
destination = '/media/ExtHDD01/Dataset/paired_images/imorphics/full/cartilages/'

txt_list = dict()
txt_list['00'] = sorted(glob.glob(os.path.join(root, '*/V00/*.txt')))
txt_list['01'] = sorted(glob.glob(os.path.join(root, '*/V01/*.txt')))

ratio = 1

for VER in ['00', '01']:
    v_list = txt_list[VER]
    for i in range(len(v_list))[:]:
        print(i)
        masks = load_imorphics_mask(v_list[i], ratio=ratio)

        original_name = v_list[i].split('/')[-1].split('_')
        new_name = original_name[0] + '_' + VER + '_' + original_name[5] + '.tif'  # (9XXXXXXX_VER_SIDE.tif)

        tiff.imsave(os.path.join(destination, new_name), masks.astype(np.uint8))
