import pandas as pd
from utils.oai_unzip import unzip_selected
from dotenv import load_dotenv
import os
load_dotenv('.env')


def dcm_2_npys(root_folder, data_name):
    dcm_folder = root_folder + data_name
    npy_folder = dcm_folder + 'Npy/'
    from utils.oai_unzip import dcm_to_npys_and_metas, meta_process
    dcm_to_npys_and_metas(source=dcm_folder,
                          destination=npy_folder,
                          metas=['ImagePositionPatient', 'SliceLocation'],
                          cohorts=['0.E.1', '0.C.2'])
    meta = meta_process(meta=pd.read_csv(npy_folder + 'meta.csv'))
    return meta


if __name__ == '__main__':
    zip00m = os.environ.get('zip00m')#'/media/ghc/GHc_data1/OAI/OAI_raw/OAIBaselineImages/results/00m.zip'
    source = os.environ.get('source')#'/media/ghc/GHc_data1/OAI/OAI_extracted/'
    data_name = 'OAI00eff0_test/'

    # unzip dicom files for the zip file
    unzip_selected(df=pd.read_csv('OAI00eff0.csv').iloc[:100,:],
                   zipname=zip00m,
                   destination=source + data_name)

    # convert the images from the dicom to .npy
    dcm_2_npys(source, data_name)