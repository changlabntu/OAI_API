import os, glob, time
import multiprocessing
import subprocess
import pydicom
import pandas as pd
import numpy as np
from PIL import Image
import shutil
import zipfile


def zip_content_list(zipfile):
    # create the list of zip file content in a txt list
    os.system('unzip -Z1 ' + zipfile + ' "*/*/*/*/" > ' + zipfile.split('.zip')[0] + '.txt')
    # read the text file to pandas
    list_of_folders = pd.read_csv(zipfile.split('.zip')[0] + '.txt', header=None)
    list_of_folders = list_of_folders.rename(columns={0: 'folders'})
    return list_of_folders


def scan_zip_locate_sequences_par(zipfile):
    """
    use scan_zip_locate_sequences in parallel
    :return:
    """
    list_of_folders = zip_content_list(zipfile)

    n_worker = 20
    range_list = np.array_split(range(list_of_folders.shape[0]), n_worker)

    workers = []
    for i in range(n_worker):
        workers.append(multiprocessing.Process(target=scan_zip_locate_sequences, args=(range_list[i], i, list_of_folders)))

    for i in range(n_worker):
        workers[i].start()

    for i in range(n_worker):
        workers[i].join()

    list_of_csv = [pd.read_csv(str(i)+'.csv') for i in range(n_worker)]
    total = pd.concat(list_of_csv)
    final = pd.DataFrame({'ID': [x.split('/')[1] for x in total['0'].values], 'sequences': total['0'], 'folders': total['1']})
    final.to_csv('extracted_oai_info/' + zipfile.split('/')[-1].split('.')[0] + '.csv', index=False)
    return 0


def scan_zip_locate_sequences(irange, n_worker, list_of_folders):
    """
    open the first slice in every subfolder of the OAI zip file to find out the type of sequence (SeriesDescription)
    :param irange:
    :param n_worker:
    :return:
    """
    dir_name = 'temp' + str(n_worker) + '/'
    found = []
    for i in irange:
        tini = time.time()
        if os.path.isdir(dir_name):
            shutil.rmtree(dir_name)
        os.mkdir(dir_name)

        sub_folder = list_of_folders['folders'][i]
        for z in range(1, 200):
            try:
                subprocess.run(['unzip', '-j', zipfile, sub_folder + str(z).zfill(3), '-d', dir_name])
                found_series_description = pydicom.read_file(glob.glob(dir_name + '*')[0]).SeriesDescription
                break
            except:
                print(str(z) + '  not found')

        list_of_folders.loc[list_of_folders['folders']
                            == sub_folder, 'SeriesDescription'] = found_series_description
        found.append((sub_folder,  found_series_description))
        print(str(i) + '  ' + found_series_description + ' ' + str(time.time() - tini))

    df = pd.DataFrame(found)
    df.to_csv(str(n_worker) + '.csv')
    return found


def unzip_selected(df, zipname, destination):
    print('Preparing archive...')
    archive = zipfile.ZipFile(zipname)
    print('Done...')
    for i in range(df.shape[0]):
        folder = df.iloc[i]['folders']
        print(folder)
        n = 1
        while True:
            try:
                archive.extract(folder+str(n).zfill(3),  destination)
                n = n + 1
            except:
                break


def find_folders_by_id_and_sequence(df, path_of_sequences):
    """
    find the folder .zip file by patient ID and MRI sequence
    """
    folders = []
    for i in range(df.shape[0]):
        ID, sequences = df.iloc[i][['ID', 'sequences']]
        folders.append(path_of_sequences.loc[(path_of_sequences['ID'] == ID) & (path_of_sequences['sequences'] == sequences), 'folders'].values[0])
    return folders


def dcm_to_npys_and_metas(source, destination, metas, cohorts):
    """
    extract npys from dcms and record meta
    """
    if not os.path.isdir(destination):
        os.mkdir(destination)

    folder_list = []
    for c in cohorts:
        folder_list = folder_list + sorted(glob.glob(source + c + '/*/*/*'))

    dcm_meta = []
    for f in folder_list:
        dcm_list = glob.glob(f+'/*')
        dcm_list.sort()

        # find ID and sequence and make folders if don't exist
        ID = f.split('/')[-3]
        sequence = pydicom.read_file(dcm_list[0]).SeriesDescription
        if not os.path.isdir(destination + sequence + '/'):
            os.mkdir(destination + sequence + '/')
        if not os.path.isdir(destination + sequence + '/' + ID + '/'):
            os.mkdir(destination + sequence + '/' + ID + '/')

        for d in dcm_list:
            dcm = pydicom.read_file(d)
            npyname = destination + sequence + '/' + ID + '/' + d.split('/')[-1]
            np.save(npyname + '.npy', dcm.pixel_array)
            meta = [sequence + '/' + ID + '/' + d.split('/')[-1]]
            for m in metas:
                meta = meta + [getattr(dcm, m)]
            dcm_meta.append(meta)
    dcm_meta = pd.DataFrame(dcm_meta, columns=['filename']+metas)
    dcm_meta.to_csv(destination + 'meta.csv', index=False)


def meta_process(meta):
    """
    process the meta data
    """
    meta['ID'] = [x.split('/')[1] for x in meta['filename']]
    meta['series'] = [x.split('/')[0].split('_')[3] for x in meta['filename']]
    meta['slice'] = [int(x.split('/')[2]) for x in meta['filename']]
    meta['sequences'] = [x.split('/')[0].split('_')[2] for x in meta['filename']]
    meta['side'] = [x.split('/')[0].split('_')[3] for x in meta['filename']]
    return meta


if __name__ == 'main':
    """
    locate pathes of MRI sequences and save as a table of [ID, sequences, folders] if not exist
    """
    zipfile = '/media/ghc/GHc_data1/OAI_raw/OAI12MonthImages/results/12m.zip'
    scan_zip_locate_sequences_par(zipfile)
