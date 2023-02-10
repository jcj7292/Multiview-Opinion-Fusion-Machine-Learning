import multiprocessing
from multiprocessing import Pool
import numpy as np
import os
import pandas as pd
import random
from tqdm import tqdm
from pymatgen.core.structure import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.analysis.diffraction.tem import TEMCalculator


def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

beam_direction_round = [[0,0,1]]
directions = []
for i in range(6):
    for j in range(6):
        for k in range(6):
            # print([i,j,k])
            angles = []
            for bd in beam_direction_round:
                angle = angle_between(bd, [i,j,k])
                angles.append(angle)
            if np.nanmin(np.asarray(angles)) > 1e-3:
                beam_direction_round.append([i,j,k])

beam_direction_round = np.asarray(beam_direction_round)


def parallel_ED(root_dir, save_dir, fnames, sym_thresh, beam_direction_i):
    process_name = multiprocessing.current_process().name
    ed_simulator = TEMCalculator(beam_direction=(beam_direction_round[beam_direction_i,0], beam_direction_round[beam_direction_i,1], beam_direction_round[beam_direction_i,2]))
    for filename in tqdm(fnames):
        cifpath = os.path.join(root_dir, 'MP_cifs', filename+'.cif')
        assert os.path.isfile(cifpath)
        cif_struct = Structure.from_file(cifpath)
        sga = SpacegroupAnalyzer(cif_struct, symprec=sym_thresh)
        # # Primitive
        # primitive_struct = sga.get_primitive_standard_structure()
        # primitive_pattern = ed_simulator.get_pattern(primitive_struct)
        # primitive_pattern.to_csv(os.path.join(save_dir, filename+'ED_prim100.csv'))
        # Conventional
        conventional_struct = sga.get_conventional_standard_structure()
        conventional_pattern = ed_simulator.get_pattern(conventional_struct)
        conventional_pattern.to_csv(os.path.join(save_dir, filename+'ED_conv.csv'))
    print('Process {} is done..'.format(process_name), flush=True)


def compute_ED(root_dir, beam_direction_i):
    # read all MPdata filenames
    assert os.path.isfile(root_dir + 'file_id.csv')
    MPdata = pd.read_csv(root_dir + 'file_id.csv', sep=';', header=0, index_col=None)
    filenames = MPdata['material_id'].values.tolist()
    
    random.shuffle(filenames)
    save_dir = os.path.join(root_dir, 'selected2/ED_simulated_'+str(beam_direction_round[beam_direction_i,0])+'_'+str(beam_direction_round[beam_direction_i,1])+'_'+str(beam_direction_round[beam_direction_i,2]))
    
    if os.path.exists(save_dir):
        _ = input("{} already exists, Enter to continue, ^C to terminate..".format(save_dir))
    else:
        os.mkdir(save_dir)
        
    sym_thresh = 0.1
    nworkers = multiprocessing.cpu_count()
    print('total size: {}, parallel computing on {} workers..'.format(len(filenames), nworkers))
    
    pool = Pool(processes=nworkers)
    file_split = np.array_split(filenames, nworkers)
    args = [(root_dir, save_dir, fnames, sym_thresh, beam_direction_i) for fnames in file_split]
    pool.starmap_async(parallel_ED, args)
    pool.close()
    pool.join()
    print('all jobs done, pool closed..')


if __name__ == '__main__':
    root_dir = '...'
    for i in range(1,len(beam_direction_round)):
        
        save_dir = os.path.join(root_dir, 'selected/ED_simulated_'+str(beam_direction_round[i,0])+'_'+str(beam_direction_round[i,1])+'_'+str(beam_direction_round[i,2]))   
        if os.path.exists(save_dir):
            continue
        
        compute_ED(root_dir, i)

