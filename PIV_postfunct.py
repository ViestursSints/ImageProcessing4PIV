import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signl
from matplotlib.colors import ListedColormap, LinearSegmentedColormap, Normalize
import os
import shutil
import re
from dataclasses import dataclass, field
import wes
import csv
from csv import DictReader
from scipy.optimize import curve_fit
import cv2 as cv

def calculate_total_velocity(u, v):
    return np.sqrt(u**2 + v**2)

def etaOstwald(gamma_p, K = 0.027816, n = 0.311958):
    eta = K*(gamma_p**(n-1));
    return eta

def lin_end(za, up_edge, downslope):
    ux = up_edge + downslope*za;
    return ux

def tauOstwald_og(gamma_p, experiment):
    if experiment.date == '4oct23' or \
          experiment.date == '22nov23_2' or \
              experiment.date == '22nov23_3' or \
                  experiment.date == '22nov23_4' or \
                    experiment.date == '25feb26' or \
                        experiment.date == '25feb26_2' or \
                            experiment.date == '25feb26_3' or \
                                experiment.date == '27feb26' or \
                                    experiment.date == '27feb26_2' or \
                                        experiment.date == '27feb26_3' or \
                                            experiment.date == '27feb26_4' or \
                                                experiment.date == '27feb26_5' or \
                                                    experiment.date == '27feb26_6' or \
                                                        experiment.date == '19feb26' or \
                                                            experiment.date == '28apr26' or \
                                                                experiment.date == '28apr26_2':
        K = 0.02781; n = 0.311958
    elif experiment.date == '6oct23':
        K = 0.036782; n = 0.329076
    elif experiment.date == '10oct23':
        K = 0.015601; n = 0.506976
    elif experiment.date == '11oct23' or \
        experiment.date == '9jan26' or \
        experiment.date == '9jan26_2' or \
        experiment.date == '9jan26_3' or \
           experiment.date == '9jan26_4' or \
           experiment.date == '9jan26_5' or \
             experiment.date == '17oct25' :
        tau = 8.9e-4*gamma_p
        #tau = 9.3e-4*gamma_p
    elif experiment.date == '11oct23X':
        tau = 0.93e-3*gamma_p
    elif experiment.date == '13mar24' or \
        experiment.date == '23may24_1' or \
            experiment.date == '23may24_2':
        K = 0.005117; n = 0.571541
    elif experiment.date == '28mar24':
        K = 0.02781; n = 0.311958
    if experiment.date != '11oct23' and experiment.date != '11oct23X' and \
        experiment.date != '9jan26' and experiment.date != '9jan26_2' and \
        experiment.date != '9jan26_3' and experiment.date != '9jan26_4' and \
            experiment.date != '17oct25' and experiment.date != '9jan26_5':
        tau = K*(gamma_p**n);
    return tau

def tauOstwald(gamma_p, experiment):

    date = experiment.date

    # ---- Newtonian / linear cases: tau = mu * gamma_p ----
    LINEAR_MODELS = {
        # Water
        '11oct23': 8.9e-4,
        '9jan26': 8.9e-4,
        '9jan26_2': 8.9e-4,
        '9jan26_3': 8.9e-4,
        '9jan26_4': 8.9e-4,
        '9jan26_5': 8.9e-4,
        '17oct25': 8.9e-4,
        '10jul26': 8.9e-4,
        '10jul26_2': 8.9e-4,
        '10jul26_3': 8.9e-4,
        '10jul26_4': 8.9e-4,
        # DMEM + FBS Literature
        '11oct23X': 0.93e-3,
    }

    # ---- Ostwald power-law cases: tau = K * gamma_p^n ----
    POWER_LAW_MODELS = {
        # DMEM + 10% FBS
        '4oct23':     (0.02781, 0.311958),
        '22nov23_2':  (0.02781, 0.311958),
        '22nov23_3':  (0.02781, 0.311958),
        '22nov23_4':  (0.02781, 0.311958),
        '25feb26':    (0.02781, 0.311958),
        '25feb26_2':  (0.02781, 0.311958),
        '25feb26_3':  (0.02781, 0.311958),
        '27feb26':    (0.02781, 0.311958),
        '27feb26_2':  (0.02781, 0.311958),
        '27feb26_3':  (0.02781, 0.311958),
        '27feb26_4':  (0.02781, 0.311958),
        '27feb26_5':  (0.02781, 0.311958),
        '27feb26_6':  (0.02781, 0.311958),
        '19feb26':    (0.02781, 0.311958),
        '19feb26_2':    (0.02781, 0.311958),
        '19feb26_3':    (0.02781, 0.311958),
        '28apr26':    (0.02781, 0.311958),
        '28apr26_2':  (0.02781, 0.311958),
        '28apr26_3':  (0.02781, 0.311958),
        '28apr26_4':  (0.02781, 0.311958),
        '28apr26_5':  (0.02781, 0.311958),
        '28apr26_6':  (0.02781, 0.311958),
        '6may26':  (0.02781, 0.311958),
        '6may26_2':  (0.02781, 0.311958),
        '6may26_3':  (0.02781, 0.311958),

        # DMEM + 10% FBS
        '6oct23':     (0.036782, 0.329076),
        # DMEM + 10% FBS
        '10oct23':    (0.015601, 0.506976),

        # DMEM + 10% FBS
        '13mar24':    (0.005117, 0.571541),
        '23may24_1':  (0.005117, 0.571541),
        '23may24_2':  (0.005117, 0.571541),
        # DMEM + 10% FBS
        '28mar24':    (0.02781, 0.311958),
    }

    # ---- Compute tau ----
    if date in LINEAR_MODELS:
        mu = LINEAR_MODELS[date]
        return mu * gamma_p

    elif date in POWER_LAW_MODELS:
        K, n = POWER_LAW_MODELS[date]
        return K * (gamma_p ** n)

    else:
        raise ValueError(f"Unknown experiment date: {date}")

def array_v_gradients(a1, a2, dz):
    du_dz = abs(a1[:, 2] - a2[:, 2]) / dz
    dv_dz = abs(a1[:, 3] - a2[:, 3]) / dz
    return du_dz, dv_dz

def slope_selector(experiment):
    endslope4 = [17, 20]; endslope6 = [15, 18];
    endslope10 = [17, 20]; endslope11 = [24, 28]; endslope13 = [21, 24];
    if experiment.date == '4oct23':
        z1 = endslope4[0]; z2 = endslope4[1];
    elif experiment.date == '6oct23':
        z1 = endslope6[0]; z2 = endslope6[1];
    elif experiment.date == '10oct23':
        z1 = endslope10[0]; z2 = endslope10[1];
    elif experiment.date == '11oct23':
        z1 = endslope11[0]; z2 = endslope11[1];
    elif experiment.date == '13mar24':
        z1 = endslope13[0]; z2 = endslope13[1];
    return z1, z2

def read_piv_output_v(exp_folder, slice_n, n, piv_res = 32):
    column_names = ["x", "y", "u", "v", "flags", "mask"]

    columns = {name: [] for name in column_names}

    file_path = exp_folder + "/PIV_output/OpenPIV_results_" + \
        str(int(piv_res)) + "_slice_" + \
          str(slice_n).zfill(3) + "/field_A" + str(n).zfill(4) + ".txt"

    with open(file_path, 'r') as file: 
        header_line = file.readline().strip()     
        for line in file:
            values = line.strip().split('\t')
            for name, value in zip(column_names, values):
                columns[name].append(float(value))

    return columns

def read_piv_output(exp_folder, slice_n, n, piv_res=32):
    column_names = ["x", "y", "u", "v", "flags", "mask"]
    columns = {name: [] for name in column_names}

    file_path = f"{exp_folder}/PIV_output/OpenPIV_results_{int(piv_res)}_slice_{str(slice_n).zfill(3)}/field_A{str(n).zfill(4)}.txt"

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Missing PIV file: {file_path}")

    with open(file_path, 'r') as file:
        header_line = file.readline().strip()
        for line in file:
            values = line.strip().split('\t')
            for name, value in zip(column_names, values):
                try:
                    columns[name].append(float(value))
                except ValueError:
                    columns[name].append(np.nan)

    for name in columns:
        arr = np.array(columns[name], dtype=np.float64)
        if not np.all(np.isfinite(arr) | np.isnan(arr)):
            raise ValueError(f"Non-finite values in column '{name}'")
        columns[name] = arr

    return columns

def read_theores(file_id):
    column_names = ["x", "var"]
    columns = {name: [] for name in column_names}

    file_path = "/home/viesturs/Fizika/PIV/theores/" + file_id + ".txt"

    with open(file_path, 'r') as file: 
        header_lines = [file.readline().strip() for _ in range(8)]    
        for line in file:
            values = line.strip().split()
            for name, value in zip(column_names, values):
                columns[name].append(float(value))

    return columns

def entire_slice_v(exp_folder, experiment, slice_n, piv_process_method, 
                 frame_n = 100, piv_res = 32, criteria = 'harsh'):

    columns = read_piv_output(exp_folder, slice_n, 0, piv_res)
    #columns = read_piv_output(exp_folder, slice_n, 1, piv_res)
    x_ve = columns['x']
    y_ve = columns['y']
    u_v = columns['u']
    v_v = columns['v']

    u_m = np.empty((np.size(x_ve),frame_n)); v_m = np.empty_like(u_m)
    u_m[:,0] = u_v; v_m[:,0] = v_v

    for i in range(0,frame_n-1):
    #for i in range(1,frame_n-1):
        columns = read_piv_output(exp_folder, slice_n, i, piv_res=piv_res)
        x_v = columns['x']
        y_v = columns['y']
        if criteria == 'kind' or criteria == 'harsh':
            u_v = list(np.where(np.array(columns['flags']) == 0.0,
                                 columns['u'], 'NaN'))
            v_v = list(np.where(np.array(columns['flags']) == 0.0,
                                 columns['v'], 'NaN'))
        if criteria == 'harsh':
            sumspeed = np.power(np.power(np.array(columns['u']),2) + \
                  np.power(np.array(columns['v']),2),0.5)
            u_v = list(np.where(sumspeed > 1e-2, u_v, 'NaN'))
            v_v = list(np.where(sumspeed > 1e-2, v_v, 'NaN'))
        if x_v == x_ve and y_v == y_ve:
            u_m[:,i] = u_v; v_m[:, i] = v_v
        else:
            print("Some are different")
    if piv_process_method == 'v':
        return np.multiply(x_ve,experiment.mmpx), np.multiply(y_ve,\
            experiment.mmpx), np.round(np.multiply(u_m,experiment.mmpx*\
            experiment.fps/1000),10), np.round(np.multiply(v_m,experiment.mmpx*experiment.fps/1000),10)
    elif piv_process_method == 'm1':
        frame_f = 5;
        fps = frame_f; #fps = 1/frame_f;
        return np.multiply(x_ve,experiment.mmpx), np.multiply(y_ve,\
            experiment.mmpx), np.round(np.multiply(u_m,experiment.mmpx*\
            fps/1000), 10), np.round(np.multiply(v_m,experiment.mmpx*fps/1000), 10)
                                                   
def entire_slice(exp_folder, experiment, slice_n, piv_process_method,
                 frame_n=100, piv_res=32, criteria='harsh'):

    columns = read_piv_output(exp_folder, slice_n, 0, piv_res)
    x_ve = columns['x']
    y_ve = columns['y']
    u_v = columns['u']
    v_v = columns['v']

    num_points = x_ve.size
    u_m = np.full((num_points, frame_n), np.nan, dtype=np.float64)
    v_m = np.full_like(u_m, np.nan)

    u_m[:, 0] = u_v
    v_m[:, 0] = v_v

    for i in range(1, frame_n):
        columns = read_piv_output(exp_folder, slice_n, i, piv_res=piv_res)
        x_v = columns['x']
        y_v = columns['y']

        if not np.array_equal(x_v, x_ve) or not np.array_equal(y_v, y_ve):
            print(f"[Frame {i}] Grid mismatch — skipping frame.")
            continue

        u_raw = columns['u']
        v_raw = columns['v']
        flags = columns['flags']

        # Base filter: flags == 0
        u_v = np.where(flags == 0.0, u_raw, np.nan)
        v_v = np.where(flags == 0.0, v_raw, np.nan)

        if criteria == 'harsh':
            sumspeed = np.sqrt(np.square(u_raw) + np.square(v_raw))
            u_v = np.where(sumspeed > 1e-2, u_v, np.nan)
            v_v = np.where(sumspeed > 1e-2, v_v, np.nan)

        u_m[:, i] = u_v
        v_m[:, i] = v_v

    scale = experiment.mmpx * experiment.fps / 1000.0

    if piv_process_method == 'v':
        return (
            x_ve * experiment.mmpx,
            y_ve * experiment.mmpx,
            np.round(u_m * scale, 10),
            np.round(v_m * scale, 10),
        )

    elif piv_process_method == 'm1':
        fps = 5  # hardcoded frame rate for 'm1'
        scale = experiment.mmpx * fps / 1000.0
        return (
            x_ve * experiment.mmpx,
            y_ve * experiment.mmpx,
            np.round(u_m * scale, 10),
            np.round(v_m * scale, 10),
        )

def slice_reducer(x_ve, y_ve, u_m, v_m,
                   outfile = False, show = False, v_focus = False, give_back = False,
                   one_slice=False):
    v_av = np.nanmean(v_m, axis = 1)
    if v_focus:
        u_av = np.zeros_like(v_av)
    else:
        u_av = np.nanmean(u_m, axis = 1)

    if show:
        plt.quiver(x_ve, y_ve, u_av, v_av)

    if outfile:
        col_format = "{:<27}" * 4 + "\n"
        with open(outfile, 'w') as file:
            for dcs in zip(x_ve, y_ve, u_av, v_av):
                file.write(col_format.format(*dcs))
    if give_back:
        x_worth = np.unique(x_ve); v_worth = np.empty_like(x_worth)
        v_worth1 = np.empty_like(x_worth); v_worth2 = np.empty_like(x_worth)
        for xi in x_worth:
            hist, bin_edges = np.histogram(v_av[np.where(x_ve == xi)][0], 15)
            peaks, properties = signl.find_peaks(hist, height=1)
            if one_slice:
                v_worth[np.where(x_worth == xi)] = v_av[np.where(x_ve == xi)][one_slice]
            else:
                v_worth1[np.where(x_worth == xi)] = 0.5*(bin_edges[peaks[
                    np.where(properties['peak_heights'] == np.max(properties[
                        'peak_heights']))]] + bin_edges[peaks[np.where(properties[
                            'peak_heights'] == np.max(properties['peak_heights']))]+1])
                v_worth2[np.where(x_worth == xi)] = np.nanmean(v_av[
                 np.where(x_ve == xi)])
                if np.sum(v_worth1[np.where(x_worth == xi)]) > \
                    np.sum(v_worth2[np.where(x_worth == xi)]):
                    #v_worth[np.where(x_worth == xi)] = v_worth1[np.where(x_worth == xi)]
                    v_worth[np.where(x_worth == xi)] = v_worth2[np.where(x_worth == xi)]
                else:
                    v_worth[np.where(x_worth == xi)] = v_worth2[np.where(x_worth == xi)]
        return x_worth, v_worth
    
def get_all_out(data_folder, experiment, piv_res, piv_process_method,
                 naming = 'v', end_slices = False):

    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    exp_folder = data_location
    slice_folder = exp_folder + "/sliced_data_" + \
        experiment.sample + '_' + str(experiment.f_rate)

    try:
          os.makedirs(slice_folder)
    except FileExistsError:
          pass
    
    for i in range(f_slice, l_slice+1):
        nf = len(os.listdir(exp_folder + "/PIV_output/OpenPIV_results_" + \
                            str(int(piv_res)) + "_slice_" + \
          str(i).zfill(3)))
        file_n = slice_folder + "/slice" + str(i).zfill(3) + ".txt"
        x_ve, y_ve, u_m, v_m = \
              entire_slice(exp_folder, experiment, slice_n=i,
                            piv_process_method = piv_process_method,
                           frame_n=nf, piv_res=piv_res)
        slice_reducer(x_ve, y_ve, u_m, v_m, outfile = file_n)

def profiler(data_folder, experiment, piv_process_method,
              piv_res = 32, save_it = False, show_it = False,
             end_slices = False, walls = False, full_test = False, one_slice = False,
             note = False, naming = 'v'):
    if experiment.f_dir == 'from_top':
        dmod = -1
    elif experiment.f_dir == 'from_bottom':
        dmod = 1

    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    slice_tot = l_slice - f_slice + 1
    x_ve, y_ve, u_m, v_m = entire_slice(data_location, experiment,
                                         slice_n=f_slice,
                                          piv_process_method = piv_process_method,
                                            frame_n=1, piv_res=piv_res)
    x_vestandart = x_ve
    x_vec_0, v_vec = slice_reducer(x_ve, y_ve, u_m, v_m, give_back = True, 
                                   one_slice=one_slice)
    if np.max(v_vec) > 1e10:
        print("Too much 1")
    if walls:
        #xi = np.where(all(x_vec > walls[0], x_vec < walls[1]))
        x_vec = x_vec_0[(x_vec_0 > walls[0]) & (x_vec_0 < walls[1])]
        v_vec = v_vec[(x_vec_0 > walls[0]) & (x_vec_0 < walls[1])]
    else:
        x_vec = x_vec_0
    x_vec = x_vec - x_vec[0]

    #v_total = np.empty([slice_tot, np.size(np.unique(x_ve))])
    v_total = np.empty([slice_tot, np.size(x_vec)])
    v_total[0,:] = dmod*v_vec

    del x_ve, y_ve, u_m, v_m

    y_vec = 1e3*experiment.y_axis
    y_vec = y_vec[f_slice - experiment.first_slice:l_slice - experiment.first_slice+1]

    for i in range (f_slice,l_slice+1):
        if full_test:
            nf = 1
        else:
            nf = len(os.listdir(data_location + \
                            "/PIV_output/OpenPIV_results_" + \
                                str(int(piv_res)) + "_slice_" + \
              str(i).zfill(3)))
        x_ve, y_ve, u_m, v_m = entire_slice(data_location,
                                            experiment, slice_n=i,
                                             piv_process_method = piv_process_method,
                                               frame_n=nf, piv_res=piv_res)
        x_vec_0, v_vec = slice_reducer(x_ve, y_ve, u_m, v_m, give_back = True, 
                                       one_slice = one_slice)
        if walls:
            x_vec = x_vec_0[(x_vec_0 > walls[0]) & (x_vec_0 < walls[1])]
            v_vec = v_vec[(x_vec_0 > walls[0]) & (x_vec_0 < walls[1])]
        else:
            x_vec = x_vec_0
        x_vec = x_vec - x_vec[0]

        if np.max(v_vec) > 1e10:
            print("Too much 2")
        v_total[i - f_slice,:] = dmod*v_vec
        v_total[v_total < 0] = 0

        del x_ve, y_ve, u_m, v_m, v_vec

    if np.max(v_total) > 1e10:
        print("Too much 3")

    Q = 0
    for i in range(len(x_vec)-1):
        for j in range(len(y_vec)-1):
            A = 1e-6*(x_vec[i+1] - x_vec[i])*(y_vec[j+1] - y_vec[j])
            Q = Q + A*v_total[j+1, i+1]

    if save_it:
        if note:
            textid = "_um-min_crosssection_" + note
        else:
            textid = "_um-min_crosssection"
        profile_file = "/more/Ex/piv" + "/" + \
              experiment.sample + "_" + str(experiment.f_rate) + textid + ".txt"
        with open(profile_file, 'w') as file:
            file.write("xy\t")
            file.write("\t".join(str(item) for item in np.transpose(x_vec)))
            file.write("\n")
            for dcs in zip(y_vec, v_total):
                file.write(str(dcs[0]))
                file.write("\t")
                file.write("\t".join(str(item) for item in dcs[1]))
                file.write("\n")
    if show_it:
        cmv = []
        #for i in range(1, 6):
        #    cmv.append(wes.wes("Zissou1", i))
        #c_nodes = [0.0, 0.1, 0.2, 0.6, 1.0]
        #cmapx = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))
        #cmapx_r = cmapx.reversed()
        for i in range(1, 11):
            cmv.append(wes.wes("AsteroidCity1", i))
        c_nodes = [0.0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        cmapx = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))
        fig1, ax1 = plt.subplots()
        contf = ax1.contourf(x_vec, y_vec, v_total, 8, vmin = 0, vmax = 3.5e-4,
                             cmap = cmapx)
        fig1.colorbar(contf, label = '$u_y$, m/s', format = '%4.2e')
        plt.xlabel('x, mm')
        plt.ylabel('z, mm')
        plt.title(experiment.sample + " " + str(experiment.f_rate) + " ul/min")
        
    return x_vec, y_vec, v_total, Q

def data_reveal(data_folder, experiment, xsection = False, slicen = False,
                 axis_of = False, axis_at = False, vmx = False, walls = False,
                 art_expo = False, scala = None, wide_scale = 1, naming = 'v'):
    
    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    if axis_of:
        yv, vaxs, dyv, dvaxs, yv_4d = one_center(data_folder, experiment,
                                      axis_at = False, walls = False)
        fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
        ax1.plot(yv, vaxs)
        fig2, ax2 = plt.subplots(figsize=(10,7), dpi=120)
        ax2.plot(yv_4d, dvaxs)
    
    if xsection:
        yv = []; vv = []
        profile_file = data_folder + "/" + \
              experiment.sample + "_" + \
                str(experiment.f_rate) + "_um-min_crosssection.txt"
        with open(profile_file, 'r') as xdata:
            opening_line = xdata.readline().strip()
            xv = [float(i) for i in opening_line.split('\t')[1:]]
            for row in xdata:
                values = row.strip().split('\t')
                yv.append(float(values[0])); vv.append([float(i) for i in values[1:]])
        vvx = np.empty((len(vv), np.size(vv[0])))
        for i in range(0,len(vv)-1):
            vvx[i,:] = vv[i]
        np.nan_to_num(vv, nan=0.0)
        cmv = []
        for i in range(1, 11):
            cmv.append(wes.wes("AsteroidCity1", i))
        c_nodes = [0.0, 0.03, 0.06, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
        cmapx = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))
        fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
        contf = ax1.contourf(xv, yv, vv, 8, vmin = 0, vmax = 3.5e-4,
                             cmap = cmapx)
        if walls:
            ax1.set_xlim(walls)
        cbar = fig1.colorbar(contf, format = '%4.2e')
        cbar.set_label(label = '$u_y$, m/s', size=14)
        cbar.ax.tick_params(labelsize=14)
        plt.xlabel('x, mm', fontsize=18)
        plt.ylabel('z, mm', fontsize=18)
        plt.title(experiment.sample + " " + str(experiment.f_rate) + " ul/min",
                  fontsize=18)
        ax1.tick_params(labelsize=14)
        ax1.set_aspect('equal', adjustable='box')

    if slicen:
        font_official = {'fontname':'Liberation Sans'}
        column_names = ["x", "y", "u", "v"]
        columns = {name: [] for name in column_names}
        if naming == 'v':
            slice_folder = data_folder + '/' + experiment.date + "/sliced_data_" + \
            experiment.sample + '_' + str(experiment.f_rate)
        elif naming == 'm1':
            slice_folder = data_folder + "/sliced_data_" + \
            experiment.sample + '_' + str(experiment.f_rate)
        file_n = slice_folder + "/slice" + str(slicen).zfill(3) + ".txt"
        with open(file_n, 'r') as the_slice:
            for line in the_slice:
                values = line.strip().split()
                for name, value in zip(column_names, values):
                    columns[name].append(float(value))
        cmv = []
        im_scale = experiment.mmpx
        for i in range(1, 6):
            cmv.append(wes.wes("Zissou1", i))
        c_nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmapx2 = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))
        Cm = np.hypot(columns['u'], columns['v'])
        fig1, ax1 = plt.subplots(figsize=(3.5,3), dpi=300)
        if naming == 'v':
            im = plt.imread(data_folder + "/" + experiment.date + \
                        "/preprocess/slice_" + \
                            str(experiment.first_slice).zfill(3) + "/000.png")
        elif naming == 'm1':
            im = plt.imread(data_location + "/run" + str(slicen) + \
                                 "/2 Filtered Images/Frame_001.tiff")
            imo = np.full_like(im, 0, dtype=np.uint8)
            im = cv.normalize(im, imo, 0, 255, cv.NORM_MINMAX)
            im = cv.cvtColor(im.astype("uint8"), cv.COLOR_GRAY2RGB)
            #im = plt.imread(data_location + "/run" + str(slicen) + \
            #                     "/Im_tracerlines.jpg")
        im_art = ax1.imshow(cv.flip(im,0), origin = 'lower')
        im_art.set_extent(np.array(im_art.get_extent())*im_scale)
        ax1.autoscale(False)
        qpl = plt.quiver(columns['x'], columns['y'], columns['u'], columns['v'], Cm,
                          cmap = cmapx2)
        plt.xlabel('y, mm', fontsize=10, **font_official)
        plt.ylabel('x, mm', fontsize=10, **font_official)
        #plt.title(experiment.sample + " " + str(experiment.f_rate) + " ul/min",
        #          fontsize=18)
        ax1.set_aspect('equal', adjustable='box')
        ax1.tick_params(labelsize=10, labelfontfamily = 'Liberation Sans')
        cbar = fig1.colorbar(qpl, label = '$u_x$, m/s', format = '%4.0e')
        cbar.set_label(label = '$u_x$, m/s', size=12, **font_official)
        cbar.ax.tick_params(labelsize=10, labelfontfamily = 'Liberation Sans')
        #plt.clim(False, vmx)

    if art_expo:
        font_official = {'fontname':'Liberation Sans'}
        column_names = ["x", "y", "u", "v"]
        columns = {name: [] for name in column_names}
        exp_folder = data_location
        #slice_folder = exp_folder + "/sliced_data_" + \
        #    experiment.sample + '_' + str(experiment.f_rate)
        if naming == 'v':
            slice_folder = data_folder + '/' + experiment.date + "/sliced_data_" + \
            experiment.sample + '_' + str(experiment.f_rate)
        elif naming == 'm1':
            slice_folder = data_folder + "/sliced_data_" + \
            experiment.sample + '_' + str(experiment.f_rate)
        file_n = slice_folder + "/slice" + str(art_expo).zfill(3) + ".txt"
        with open(file_n, 'r') as the_slice:
            for line in the_slice:
                values = line.strip().split()
                for name, value in zip(column_names, values):
                    columns[name].append(float(value))
        im_scale = experiment.mmpx
        Cm = np.hypot(columns['u'], columns['v'])
        fig1, ax1 = plt.subplots(figsize=(3.5,3), dpi=300)
        if naming == 'v':
            im = plt.imread(data_folder + "/" + experiment.date + \
                        "/preprocess/slice_" + \
                            str(experiment.first_slice).zfill(3) + "/000.png")
        elif naming == 'm1':
            #im = plt.imread(data_location + "/run" + str(slicen) + \
            #                     "/2 Filtered Images/Frame_1.tiff")
            im = plt.imread(data_location + "/run" + str(art_expo) + \
                                 "/Im_tracerlines.jpg")
        im_art = ax1.imshow(cv.flip(1 - im, 0), origin='lower')
        im_art.set_extent(np.array(im_art.get_extent())*im_scale)
        ax1.autoscale(False)
        qpl = plt.quiver(columns['x'], columns['y'], columns['u'], columns['v'], Cm,
                          cmap = 'coolwarm', scale = scala, scale_units='xy',
                          width=wide_scale)
        plt.xlabel('y, mm', fontsize=10, **font_official)
        plt.ylabel('x, mm', fontsize=10, **font_official)
        plt.title(experiment.sample + " " + str(experiment.f_rate) + " ul/min",
                  fontsize=20)
        ax1.set_xlim(0.12, 0.33); ax1.set_ylim(0.2, 0.41)
        ax1.set_aspect('equal', adjustable='box')
        ax1.tick_params(labelsize=10, labelfontfamily = 'Liberation Sans')
        cbar = fig1.colorbar(qpl, label = '$u_x$, m/s', format = '%4.1e', location='left')
        cbar.set_label(label = '$u_x$, m/s', size=12, **font_official)
        cbar.ax.tick_params(labelsize=10, labelfontfamily = 'Liberation Sans')

    #return columns

def cell_emphasis(data_folder, experiment, slice_1, last_tick,
                 axis_at = False, vmx = False, wallsx = False, wallsy = False,
                 scala = None, wide_scale = 1, naming = 'v', normz = False,
                  cnorm = 5e-5):

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    font_official = {'fontname':'Liberation Sans'}
    column_names = ["x", "y", "u", "v"]
    columns = {name: [] for name in column_names}
    exp_folder = data_location
    if naming == 'v':
        slice_folder = data_folder + '/' + experiment.date + "/sliced_data_" + \
        experiment.sample + '_' + str(experiment.f_rate)
    elif naming == 'm1':
        slice_folder = data_folder + "/sliced_data_" + \
        experiment.sample + '_' + str(experiment.f_rate)
    im_scale = experiment.mmpx
    
    fig1, ax1 = plt.subplots(1, 4, figsize=(11,6), dpi=450)

    cmv = []
    for i in range(1, 6):
            cmv.append(wes.wes("Zissou1", i))
    c_nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
    cmapx2 = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))

    for art_expo in range(slice_1, slice_1 + 4):
        file_n = slice_folder + "/slice" + str(art_expo).zfill(3) + ".txt"
        with open(file_n, 'r') as the_slice:
            for line in the_slice:
                values = line.strip().split()
                for name, value in zip(column_names, values):
                    columns[name].append(float(value))
        
        Cm = np.hypot(columns['u'], columns['v'])
        
        if naming == 'v':
            im = plt.imread(data_folder + "/" + experiment.date + \
                        "/preprocess/slice_" + \
                            str(experiment.first_slice).zfill(3) + "/000.png")
        elif naming == 'm1':
            #im = plt.imread(data_location + "/run" + str(slicen) + \
            #                     "/2 Filtered Images/Frame_1.tiff")
            im = plt.imread(data_location + "/run" + str(art_expo) + \
                                 "/Im_tracerlines.jpg")
        im_art = ax1[art_expo-slice_1].imshow(cv.flip(1 - im, 0), origin='lower')
        im_art.set_extent(np.array(im_art.get_extent())*im_scale)
        ax1[art_expo-slice_1].autoscale(False)
        if normz:
            u = []; v = [];
            for i in range(0,len(columns['u'])):
                u.append(columns['u'][i]/Cm[i]);
                v.append(columns['v'][i]/Cm[i]);
            #u = columns['u']/abs(columns['u']); v = columns['v']/abs(columns['v'])
        else:
            u = columns['u']; v = columns['v']
        cnormv = Normalize(vmin=0, vmax=cnorm)
        qpl = ax1[art_expo-slice_1].quiver(columns['x'], columns['y'], u,
                                              v, Cm,
                          cmap = cmapx2, scale = scala, scale_units='xy',
                          width=wide_scale)
        #qpl = ax1[art_expo-slice_1].quiver(columns['x'], columns['y'], u,
        #                                      v, Cm,
        #                  cmap = cmapx2, scale = scala, scale_units='xy',
        #                  width=wide_scale, norm=cnormv)
        #qpl = ax1[art_expo-slice_1].quiver(columns['x'], columns['y'], u,
        #                                      v, Cm,
        #                  cmap = cmapx2, scale = scala, scale_units='xy',
        #                  width=wide_scale)
        ax1[art_expo-slice_1].set_xlabel('y, mm', fontsize=12, **font_official)
        if art_expo == slice_1:
            ax1[art_expo-slice_1].set_ylabel('x, mm', fontsize=12, **font_official)
        if art_expo != slice_1:
            qpl.axes.yaxis.set_ticklabels([])
        thiz = 1e6*(experiment.y_axis[art_expo] - experiment.y_axis[slice_1])
        ax1[art_expo-slice_1].set_title("z="+ '%1.0f' % thiz + ' um',
                  fontsize=12)
        ax1[art_expo-slice_1].set_xlim(wallsx[0], wallsx[1])
        ax1[art_expo-slice_1].set_ylim(wallsy[0], wallsy[1])
        ax1[art_expo-slice_1].set_aspect('equal', adjustable='box')
        ax1[art_expo-slice_1].tick_params(labelsize=12,
                                           labelfontfamily = 'Liberation Sans')
    cbar = fig1.colorbar(qpl, ax=ax1[:4], shrink = 0.6,
                          label = '$u_x$, m/s', format = '%4.1e', location='bottom',
                          ticks = np.linspace(0, last_tick, 4))
    cbar.set_label(label = '$u_x$, m/s', size=12, **font_official)
    cbar.ax.tick_params(labelsize=12, labelfontfamily = 'Liberation Sans')

def by_slice(data_folder, experiment, slice, frame_n = 100,
              naming='m1', piv_process_method = 'm1', piv_res=32):

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    out_location = data_location + "/vfields/slice_" + str(slice).zfill(3)
    try:
        os.makedirs(out_location)
    except FileExistsError:
        pass

    x_ve, y_ve, u_m, v_m = entire_slice(data_location, experiment,
                                         slice_n=slice,
                                          piv_process_method = piv_process_method,
                                            frame_n=frame_n, piv_res=piv_res)
    for i in range(0,frame_n-1):
        cmv = []
        im_scale = experiment.mmpx
        for j in range(1, 6):
            cmv.append(wes.wes("Zissou1", j))
        c_nodes = [0.0, 0.25, 0.5, 0.75, 1.0]
        cmapx2 = LinearSegmentedColormap.from_list("cmapx", list(zip(c_nodes, cmv)))
        Cm = np.hypot(u_m[:,i], v_m[:,i])
        fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
        if naming == 'v':
            im = plt.imread(data_folder + "/" + experiment.date + \
                        "/preprocess/slice_" + \
                            str(experiment.first_slice).zfill(3) + "/000.png")
        elif naming == 'm1':
            im = plt.imread(data_location + "/run" + str(slice) + \
                                 "/Im_tracerlines.jpg")
        im_art = ax1.imshow(cv.flip(im,0), origin = 'lower')
        im_art.set_extent(np.array(im_art.get_extent())*im_scale)
        ax1.autoscale(False)
        thenorm = Normalize(vmin=0, vmax=np.max([np.max(u_m),np.max(v_m)]))
        qpl = plt.quiver(x_ve, y_ve, u_m[:,i], v_m[:,i], Cm,
                          cmap = cmapx2, norm = thenorm)
        plt.xlabel('x, mm', fontsize=18)
        plt.ylabel('y, mm', fontsize=18)
        plt.title(experiment.sample + " " + str(experiment.f_rate) + " ul/min",
                  fontsize=18)
        ax1.set_aspect('equal', adjustable='box')
        ax1.tick_params(labelsize=14)
        cbar = fig1.colorbar(qpl, label = '$u_y$, m/s', format = '%4.2e',
                             ticks = np.linspace(0,
                                                 np.max([np.max(u_m),np.max(v_m)]),
                                                 6))
        cbar.set_label(label = '$u_y$, m/s', size=14)
        cbar.ax.tick_params(labelsize=14)
        plt.savefig(out_location + "/" + str(i).zfill(3) + ".png",
                     bbox_inches='tight')
        plt.close(fig1)  # Close the figure to free up memory


def write_to_vtk_og(data_folder, experiment, piv_process_method, naming,
                  end_slices = False, walls = False, piv_res = 32):
    
    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder
    
    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice
    z_coordinates = 1e3*experiment.y_axis
    z_coordinates = \
        z_coordinates[f_slice - experiment.first_slice:l_slice - experiment.first_slice+1]

    data_arrays = []
    for i in range (f_slice,l_slice+1):
        nf = len(os.listdir(data_location + \
                            "/PIV_output/OpenPIV_results_" + \
                                 str(int(piv_res)) + "_slice_" + \
              str(i).zfill(3)))
        x_ve, y_ve, u_m, v_m = entire_slice(data_location,
                                            experiment, slice_n=i,
                                             piv_process_method = piv_process_method,
                                               frame_n=nf, piv_res=piv_res)
        v_av = np.nanmean(v_m, axis = 1); u_av = np.nanmean(u_m, axis = 1)
        x_ve0 = x_ve
        if walls:
            x_ve = x_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            y_ve = y_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            v_av = v_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            u_av = u_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
        data_array = np.column_stack((x_ve, y_ve, v_av, u_av))
        data_arrays.append(data_array)

        if 'nan' in v_av:
            for j in [i for i in range(len(v_av)) if v_av[i] == 'nan']:
                v_av[j] = (v_av[j-1] + v_av[j+1])/2

        if 'nan' in u_av:
            for j in [i for i in range(len(u_av)) if u_av[i] == 'nan']:
                u_av[j] = (u_av[j-1] + u_av[j+1])/2

    output_file_path = "/more/Ex/piv" + "/" + \
              experiment.sample + "_" + str(experiment.f_rate) + ".vtk"

    # Write the data to a VTK file
    with open(output_file_path, 'w') as vtk_file:
        # Write the header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Write the points
        vtk_file.write("POINTS {} float\n".format(len(z_coordinates) * len(data_arrays[0])))
        for zi in range(0,np.size(z_coordinates)):
            z = z_coordinates[zi]
            data_array = data_arrays[zi]
            for x, y in data_array[:, :2]:
                vtk_file.write("{} {} {}\n".format(x, y, z))

        # Write the point data
        vtk_file.write("POINT_DATA {}\n".format(len(z_coordinates) * len(data_arrays[0])))
        vtk_file.write("VECTORS velocity float\n")
        for data_array in data_arrays:
            for u, v in data_array[:, 2:]:
                if abs(u) > 1e-3 or u == 'nan':
                    u = '0.0'
                elif abs(u) < 1e-10:
                    u = 0.0
                else:
                    u = format(u, '.6g')
                if abs(v) > 1e-3 or v == 'nan':
                    v = '0.0'
                elif abs(v) < 1e-10:
                    v = 0.0
                else:
                    v = format(v, '.6g')
                vtk_file.write("{} {} 0.0\n".format(v, u))

def write_to_vtk(data_folder, experiment, piv_process_method, naming,
                  end_slices = False, walls = False, piv_res = 32):
    
    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder
    
    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice
    z_all = 1e3 * experiment.y_axis

    data_arrays_all = []

    # Step 2: build data for ALL slices
    for i in range(experiment.first_slice, experiment.last_slice + 1):
        nf = len(os.listdir(
            data_location + "/PIV_output/OpenPIV_results_" +
            str(int(piv_res)) + "_slice_" + str(i).zfill(3)
        ))

        x_ve, y_ve, u_m, v_m = entire_slice(
            data_location,
            experiment,
            slice_n=i,
            piv_process_method=piv_process_method,
            frame_n=nf,
            piv_res=piv_res
        )

        v_av = np.nanmean(v_m, axis=1)
        u_av = np.nanmean(u_m, axis=1)

        x_ve0 = x_ve

        if walls:
            mask = (x_ve0 > walls[0]) & (x_ve0 < walls[1])
            x_ve = x_ve[mask]
            y_ve = y_ve[mask]
            v_av = v_av[mask]
            u_av = u_av[mask]

        data_array = np.column_stack((x_ve, y_ve, v_av, u_av))
        data_arrays_all.append(data_array)

        if 'nan' in v_av:
            for j in [i for i in range(len(v_av)) if v_av[i] == 'nan']:
                v_av[j] = (v_av[j-1] + v_av[j+1])/2

        if 'nan' in u_av:
            for j in [i for i in range(len(u_av)) if u_av[i] == 'nan']:
                u_av[j] = (u_av[j-1] + u_av[j+1])/2

    # Step 3: sort everything by z
    sort_idx = np.argsort(z_all)

    z_sorted = z_all[sort_idx]
    data_sorted = [data_arrays_all[i] for i in sort_idx]

    # Step 4: NOW apply slice selection in sorted space
    z_coordinates = z_sorted[f_slice:l_slice + 1]
    data_arrays = data_sorted[f_slice:l_slice + 1]

    output_file_path = "/more/Ex/piv" + "/" + \
              experiment.sample + "_" + str(experiment.f_rate) + ".vtk"

    # Write the data to a VTK file
    with open(output_file_path, 'w') as vtk_file:
        # Write the header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Write the points
        vtk_file.write("POINTS {} float\n".format(len(z_coordinates) * len(data_arrays[0])))
        for zi in range(0,np.size(z_coordinates)):
            z = z_coordinates[zi]
            data_array = data_arrays[zi]
            for x, y in data_array[:, :2]:
                vtk_file.write("{} {} {}\n".format(x, y, z))

        # Write the point data
        vtk_file.write("POINT_DATA {}\n".format(len(z_coordinates) * len(data_arrays[0])))
        vtk_file.write("VECTORS velocity float\n")
        for data_array in data_arrays:
            for u, v in data_array[:, 2:]:
                if abs(u) > 1e-3 or u == 'nan':
                    u = '0.0'
                elif abs(u) < 1e-10:
                    u = 0.0
                else:
                    u = format(u, '.6g')
                if abs(v) > 1e-3 or v == 'nan':
                    v = '0.0'
                elif abs(v) < 1e-10:
                    v = 0.0
                else:
                    v = format(v, '.6g')
                vtk_file.write("{} {} 0.0\n".format(v, u))

def write_to_vtk_s(data_folder, experiment, piv_process_method,
                   end_slices = False, walls = False, piv_res = 32):
    
    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice
    z_coordinates = 1e3*experiment.y_axis
    z_coordinates = \
        z_coordinates[f_slice - experiment.first_slice:l_slice - experiment.first_slice+1]

    data_arrays = []
    for i in range (f_slice,l_slice+1):
        nf = len(os.listdir(data_folder + '/' + experiment.date + \
                            "/PIV_output/OpenPIV_results_" + \
                                 str(int(piv_res)) + "_slice_" + \
              str(i).zfill(3)))
        x_ve, y_ve, u_m, v_m = entire_slice(data_folder + '/' + experiment.date,
                                            experiment, slice_n=i, 
                                            piv_process_method = piv_process_method,
                                            frame_n=nf, piv_res=piv_res)
        v_av = np.nanmean(v_m, axis = 1); u_av = np.nanmean(u_m, axis = 1)
        x_ve0 = x_ve
        if walls:
            x_ve = x_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            y_ve = y_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            v_av = v_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            u_av = u_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
        data_array = np.column_stack((x_ve, y_ve, v_av, u_av))
        data_arrays.append(data_array)

        if 'nan' in v_av:
            for j in [i for i in range(len(v_av)) if v_av[i] == 'nan']:
                v_av[j] = (v_av[j-1] + v_av[j+1])/2

        if 'nan' in u_av:
            for j in [i for i in range(len(u_av)) if u_av[i] == 'nan']:
                u_av[j] = (u_av[j-1] + u_av[j+1])/2

    output_file_path = "/more/Ex/piv" + "/" + \
              experiment.sample + "_" + str(experiment.f_rate) + "_s.vtk"

    # Write the data to a VTK file
    with open(output_file_path, 'w') as vtk_file:
        # Write the header
        vtk_file.write("# vtk DataFile Version 3.0\n")
        vtk_file.write("vtk output\n")
        vtk_file.write("ASCII\n")
        vtk_file.write("DATASET UNSTRUCTURED_GRID\n")

        # Write the points
        vtk_file.write("POINTS {} float\n".format(len(z_coordinates) * len(data_arrays[0])))
        for zi in range(0,np.size(z_coordinates)):
            z = z_coordinates[zi]
            data_array = data_arrays[zi]
            for x, y in data_array[:, :2]:
                vtk_file.write("{} {} {}\n".format(x, y, z))

        # Write the point data
        vtk_file.write("POINT_DATA {}\n".format(len(z_coordinates) * len(data_arrays[0])))
        vtk_file.write("SCALARS total_velocity float\n")
        vtk_file.write("LOOKUP_TABLE default\n")
        for data_array in data_arrays:
            for u, v in data_array[:, 2:]:
                total_velocity = calculate_total_velocity(u, v)
                vtk_file.write("{}\n".format(total_velocity))

def calculate_gradient(x, y, u, v):
    dx = np.gradient(x)
    dy = np.gradient(y)
    du_dx = np.gradient(u, axis=0) / dx[:, np.newaxis]
    dv_dy = np.gradient(v, axis=1) / dy[np.newaxis, :]
    du_dy = np.gradient(u, axis=1) / dy[np.newaxis, :]
    dv_dx = np.gradient(v, axis=0) / dx[:, np.newaxis]
    return du_dx, dv_dy, du_dy, dv_dx

def calculate_velocity_gradients_og(data_arrays, z_coordinates, experiment):
    all_gradients = []
    all_taus = []

    for i in range(len(z_coordinates)-1):
        z = z_coordinates[i+1]
        current_array = data_arrays[i+1]

        if i > 0:
            previous_array = data_arrays[i - 1]
            dz = z - z_coordinates[i - 1]

            #fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
            #cpl = plt.contourf(np.unique(previous_array[:,0]),
            #                   np.unique(previous_array[:,1]),
            #                   np.reshape())

            # Calculate velocity gradients in the z direction
            du_dz = abs(current_array[:, 2] - previous_array[:, 2]) / dz
            dv_dz = abs(current_array[:, 3] - previous_array[:, 3]) / dz

            all_gradients.append((du_dz, dv_dz))
            all_taus.append((tauOstwald(du_dz, experiment),
                              tauOstwald(dv_dz, experiment)))

        # Calculate gradient for each data array in the xy plane
        #x, y, u, v = current_array[:, 0], current_array[:, 1],
         #current_array[:, 2], current_array[:, 3]
        #du_dx, dv_dy, du_dy, dv_dx = calculate_gradient(x, y, u, v)
        #all_gradients.append((du_dx, dv_dy, du_dy, dv_dx))

    return all_gradients, all_taus

def calculate_velocity_gradients_alt(data_arrays, z_coordinates, experiment):
    all_gradients = []
    all_taus = []

    for i in range(len(z_coordinates)-2):
        z = z_coordinates[i+1]
        current_array = data_arrays[i+1]

        previous_array = data_arrays[i]
        dz1 = z - z_coordinates[i]
        next_array = data_arrays[i + 2]
        dz2 = z_coordinates[i + 2] - z

            #fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
            #cpl = plt.contourf(np.unique(previous_array[:,0]),
            #                   np.unique(previous_array[:,1]),
            #                   np.reshape())

            # Calculate velocity gradients in the z direction
        du_dz1 = abs(current_array[:, 2] - previous_array[:, 2]) / dz1
        dv_dz1 = abs(current_array[:, 3] - previous_array[:, 3]) / dz1

        du_dz2 = abs(next_array[:, 2] - current_array[:, 2]) / dz2
        dv_dz2 = abs(next_array[:, 3] - current_array[:, 3]) / dz2

        du_dz = (du_dz1 + du_dz2)/2
        dv_dz = (dv_dz1 + dv_dz2)/2

        all_gradients.append((du_dz, dv_dz))
        all_taus.append((tauOstwald(du_dz, experiment),
                          tauOstwald(dv_dz, experiment)))

        # Calculate gradient for each data array in the xy plane
        #x, y, u, v = current_array[:, 0], current_array[:, 1], current_array[:, 2],
         #current_array[:, 3]
        #du_dx, dv_dy, du_dy, dv_dx = calculate_gradient(x, y, u, v)
        #all_gradients.append((du_dx, dv_dy, du_dy, dv_dx))

    return all_gradients, all_taus

def calculate_velocity_gradients(data_arrays, z_coordinates, experiment, take):
    all_gradients = []
    all_taus = []

    if take == 'fit':
        z1, z2 = slope_selector(experiment)
        z_slope = z_coordinates[z1:z2]; z_indx = np.linspace(z1, z2-1, z2-z1);
        ue_arr = np.zeros_like(data_arrays[0][:,2]);
        s_arr = np.zeros_like(data_arrays[0][:,2])
        t_arr = np.zeros_like(data_arrays[0][:,2]);
        for j in range(np.shape(data_arrays[0][:,2])[0]):
            u_slope = []
            for zi in z_indx:
                u_slope.append(data_arrays[int(zi)][:,2][j])
            ue0 = 0.00005;
            s0 = -0.3;
            pampams, pcov = curve_fit(lin_end,
                        z_slope, u_slope, p0 = [ue0, s0])
            ue_arr[j] = pampams[0]; s_arr[j] = pampams[1]
            t_arr[j] = tauOstwald(abs(s_arr[j]), experiment)
        all_gradients = s_arr; all_taus = t_arr
    else:
        for i in range(len(z_coordinates)-2):
            current_array = data_arrays[i+1];
            previous_array = data_arrays[i]; next_array = data_arrays[i + 2]

            if take == 'og':
                dz = z_coordinates[i + 2] - z_coordinates[i]
                du_dz, dv_dz = array_v_gradients(current_array, previous_array, dz)
                all_gradients.append((du_dz, dv_dz))
                all_taus.append((tauOstwald(du_dz, experiment),
                              tauOstwald(dv_dz, experiment)))
            elif take == 'alt':
                dz1 = z_coordinates[i + 1] - z_coordinates[i]
                dz2 = z_coordinates[i + 2] - z_coordinates[i + 1]
                du_dz1, dv_dz1 = array_v_gradients(current_array, previous_array, dz1)
                du_dz2, dv_dz2 = array_v_gradients(next_array, current_array, dz2)
                du_dz = (du_dz1 + du_dz2)/2
                dv_dz = (dv_dz1 + dv_dz2)/2
                all_gradients.append((du_dz, dv_dz))
                all_taus.append((tauOstwald(du_dz, experiment),
                          tauOstwald(dv_dz, experiment)))

    return all_gradients, all_taus

def v_gradients(data_folder, experiment, take, piv_process_method, naming,
                 end_slices = False, walls = False, piv_res = 32):
    
    if end_slices:
        f_slice, l_slice = end_slices
    else:
        f_slice = experiment.first_slice; l_slice = experiment.last_slice
    z_coordinates = experiment.y_axis
    z_coordinates = \
        z_coordinates[f_slice - experiment.first_slice:l_slice - experiment.first_slice+1]

    data_arrays = []

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    for i in range (f_slice,l_slice+1):
        nf = len(os.listdir(data_location + \
                            "/PIV_output/OpenPIV_results_" + \
                                 str(int(piv_res)) + "_slice_" + \
              str(i).zfill(3)))
        x_ve, y_ve, u_m, v_m = entire_slice(data_location,
                                            experiment, slice_n=i, 
                                            piv_process_method = piv_process_method,
                                            frame_n=nf, piv_res=piv_res)
        v_av = np.nanmean(v_m, axis = 1); u_av = np.nanmean(u_m, axis = 1)
        x_ve0 = x_ve
        if walls:
            x_ve = x_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            y_ve = y_ve[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            v_av = v_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]
            u_av = u_av[(x_ve0 > walls[0]) & (x_ve0 < walls[1])]

        if 'nan' in v_av:
            for j in [i for i in range(len(v_av)) if v_av[i] == 'nan']:
                v_av[j] = (v_av[j-1] + v_av[j+1])/2

        if 'nan' in u_av:
            for j in [i for i in range(len(u_av)) if u_av[i] == 'nan']:
                u_av[j] = (u_av[j-1] + u_av[j+1])/2

        data_array = np.column_stack((x_ve, y_ve, v_av, u_av))
        data_arrays.append(data_array)

    velocity_gradients, taus = calculate_velocity_gradients(data_arrays,
                                                                 z_coordinates,
                                                                   experiment, take)

    return velocity_gradients, taus, x_ve, y_ve, z_coordinates, data_arrays

def stdev_2d(data_folder, experiment, slicen, 
             piv_process_method, naming, piv_res = 32, save_it = False):

    if naming == 'v':
        data_location = data_folder + '/' + experiment.date
    elif naming == 'm1':
        data_location = data_folder

    nf = len(os.listdir(data_location + \
                            "/PIV_output/OpenPIV_results_" + \
                                 str(int(piv_res)) + "_slice_" + \
              str(slicen).zfill(3)))
    x_ve, y_ve, u_m, v_m = entire_slice(data_location,
                                            experiment, slice_n=slicen, 
                                            piv_process_method = piv_process_method,
                                            frame_n=nf, piv_res=piv_res)
    v_std = np.nanstd(v_m, axis = 1); u_std = np.nanstd(u_m, axis = 1)
    v_me = np.nanmean(v_m, axis = 1); u_me = np.nanmean(u_m, axis = 1)
    stdv = np.power(np.power(v_std,2) + np.power(u_std,2),0.5)
    fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
    cpl = plt.contourf(np.unique(x_ve), np.unique(y_ve),
                np.reshape(stdv,(len(np.unique(y_ve)), len(np.unique(x_ve)))))
    plt.xlabel('x, mm', fontsize = 18)
    plt.ylabel('y, mm', fontsize = 18)
    cbar = fig1.colorbar(cpl, label = 'u stdev', format = '%4.2e')

    if save_it:
        textid = "_" + "slice" + str(slicen) + "_stdev"
        outfold = data_folder + "/stdevs"
        try:
            os.makedirs(outfold)
        except FileExistsError:
            pass
        profile_file = outfold + "/" + \
                experiment.sample + "_" + str(experiment.f_rate) + textid + ".txt"
        with open(profile_file, 'w') as file:
            file.write("xy\t")
            file.write("\t".join(str(item) for item in np.transpose(np.unique(x_ve))))
            file.write("\n")
            for dcs in zip(np.unique(y_ve),
             np.reshape(u_std,(len(np.unique(y_ve)), len(np.unique(x_ve))))):
                file.write(str(dcs[0]))
                file.write("\t")
                file.write("\t".join(str(item) for item in dcs[1]))
                #file.write("\t".join(str(item) for item in dcs))
                file.write("\n")

def shearstress_1d(data_folder, experiment, slices, piv_res=32):
    vg_2311_3, tau2311_3, x_2311_3, y_2311_3, za2311_3, data_arrays = v_gradients(data_folder, 
                                 experiment, piv_res=piv_res)
    all_shear_stresses = []
    for slice in slices:
        all_shear_stresses.append(tau2311_3[slice][1])

        fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
        cpl = plt.contourf(np.unique(x_2311_3), np.unique(y_2311_3),
              np.reshape(tau2311_3[slice][1],
                          (len(np.unique(y_2311_3)), len(np.unique(x_2311_3)))))
        plt.xlabel('x, mm')
        plt.ylabel('y, mm')
        fig1.colorbar(cpl, label = '$\\tau$, Pa', format = '%4.2e')

def shearstress_2d(data_folder, experiment, slices,
                    piv_res=32, walls=False, take='alt', naming = 'v'):
    piv_process_method = naming;
    vg_2311_3,tau2311_3,x_2311_3,y_2311_3,za2311_3, data_arrays = v_gradients(data_folder, 
                                 experiment, take, piv_process_method, naming,
                                   piv_res=piv_res)
    all_shear_stresses = []
    
    if take != 'fit':
        for slice in slices:
            all_shear_stresses.append(tau2311_3[slice][0])

            fig1, ax1 = plt.subplots(figsize=(10,7), dpi=120)
            cpl = plt.contourf(np.unique(x_2311_3), np.unique(y_2311_3),
                np.reshape(tau2311_3[slice][0],
                          (len(np.unique(y_2311_3)), len(np.unique(x_2311_3)))))
            plt.xlabel('x, mm', fontsize = 18)
            plt.ylabel('y, mm', fontsize = 18)
            ax1.tick_params(labelsize=14)
            if walls:
                ax1.set_xlim(walls)
            cbar = fig1.colorbar(cpl, label = '$\\tau$, Pa', format = '%4.2e')
            cbar.set_label(label = '$\\tau$, Pa', size=14)
            cbar.ax.tick_params(labelsize=14)
    else:
        all_shear_stresses = tau2311_3

    return all_shear_stresses, x_2311_3, y_2311_3

def shearstress_xprofiles(data_folder, experiment, axis_at, piv_res=32):
    vg_2311_3, tau2311_3, x_2311_3, y_2311_3, za2311_3, data_arrays = v_gradients(data_folder, 
                                 experiment, take='alt', piv_res=piv_res)
    s_profiles = []; v_profile = []
    xv = list(x_2311_3)
    x_at_axis = xv[np.where(np.array(xv) <= axis_at)[0][-1]]
    xind = xv.index(x_at_axis)
    for z in za2311_3:
        s_profiles.append(np.mean(np.reshape(tau2311_3[z][1],
                          (len(np.unique(y_2311_3)), len(np.unique(x_2311_3)))),axis=0))
        v_profile.append(s_profiles[-1][xind])
    return s_profiles, v_profile

def iso_surfs():

    return 1

def cube_stab(data_arrays, z_coordinates, experiment, take):
    all_gradients = []
    all_taus = []

    if take == 'fit':
        z1, z2 = slope_selector(experiment)
        z_slope = z_coordinates[z1:z2]; z_indx = np.linspace(z1, z2-1, z2-z1);
        ue_arr = np.zeros_like(data_arrays[0][:,2]);
        s_arr = np.zeros_like(data_arrays[0][:,2])
        t_arr = np.zeros_like(data_arrays[0][:,2]);
        for j in range(np.shape(data_arrays[0][:,2])[0]):
            u_slope = []
            for zi in z_indx:
                u_slope.append(data_arrays[int(zi)][:,2][j])
            ue0 = 0.00005;
            s0 = -0.3;
            pampams, pcov = curve_fit(lin_end,
                        z_slope, u_slope, p0 = [ue0, s0])
            ue_arr[j] = pampams[0]; s_arr[j] = pampams[1]
            t_arr[j] = tauOstwald(abs(s_arr[j]), experiment)
        all_gradients = s_arr; all_taus = t_arr
    else:
        for i in range(len(z_coordinates)-2):
            current_array = data_arrays[i+1];
            previous_array = data_arrays[i]; next_array = data_arrays[i + 2]

            if take == 'og':
                dz = z_coordinates[i + 2] - z_coordinates[i]
                du_dz, dv_dz = array_v_gradients(current_array, previous_array, dz)
                all_gradients.append((du_dz, dv_dz))
                all_taus.append((tauOstwald(du_dz, experiment),
                              tauOstwald(dv_dz, experiment)))
            elif take == 'alt':
                dz1 = z_coordinates[i + 1] - z_coordinates[i]
                dz2 = z_coordinates[i + 2] - z_coordinates[i + 1]
                du_dz1, dv_dz1 = array_v_gradients(current_array, previous_array, dz1)
                du_dz2, dv_dz2 = array_v_gradients(next_array, current_array, dz2)
                du_dz = (du_dz1 + du_dz2)/2
                dv_dz = (dv_dz1 + dv_dz2)/2
                all_gradients.append((du_dz, dv_dz))
                all_taus.append((tauOstwald(du_dz, experiment),
                          tauOstwald(dv_dz, experiment)))

    return all_gradients, all_taus

def one_center(data_folder, experiment, axis_at = False, walls = False):
    yv = []; vaxs = []; dyv = []; dvaxs = []; yv_4d = [];
    dvt = []; taxs = []; yv_4t = []
    experiment_t = experiment
    if(experiment.date == '11oct23X'):
        experiment = ex_data('11oct23', 1)
    profile_file = data_folder + "/" + \
            experiment.sample + "_" + \
            str(experiment.f_rate) + "_um-min_crosssection.txt"
    with open(profile_file, 'r') as xdata:
        opening_line = xdata.readline().strip()
        xv = [float(i) for i in opening_line.split('\t')[1:]]
        if axis_at:
            x_at_axis = xv[np.where(np.array(xv) <= axis_at)[0][-1]]
        else:
            if walls:
                x_at_axis = xv[xv <= (walls[0] + (walls[1] - walls[0])/2)][-1]
            else:
                x_at_axis = xv[len(xv)//2]
        values = xdata.readline().strip().split('\t')
        xind = xv.index(x_at_axis)
        yv.append(1e-3*float(values[0]));
        #vaxs.append(float(values[1+xind]))
        vaxs.append((float(values[xind])+float(values[1+xind])+float(values[2+xind]))/3)
        for row in xdata:
            values = row.strip().split('\t')
            yv.append(1e-3*float(values[0]));
            #vaxs.append(float(values[1+xind]))
            vaxs.append((float(values[xind])+
                         float(values[1+xind])+float(values[2+xind]))/3)
            if yv[-2] != yv[-1]:
                yv_4d.append(yv[-1])
                dyv.append(yv[-1] - yv[-2]); dvaxs.append((vaxs[-1] - vaxs[-2])/dyv[-1])
        for i in range(1,len(yv)-1):
            if(yv[i-1] != yv[i] and yv[i] != yv[i+1]):
                yv_4t.append(yv[i])
                dyv1 = yv[i] - yv[i-1]; dyv2 = yv[i+1] - yv[i]
                dv = ((abs(vaxs[i] - vaxs[i-1])/dyv1) + (abs(vaxs[i+1] - vaxs[i])/dyv2))/2
                dvt.append(dv)
                taxs.append(tauOstwald(dv, experiment_t))
    return yv, vaxs, dyv, dvaxs, yv_4d, dvt, yv_4t, taxs

def endslope(yv, vaxs, endpoints):
    yve = np.array(yv[endpoints[0]:endpoints[1]])  - yv[endpoints[0]]
    vaxe = vaxs[endpoints[0]:endpoints[1]]
    ue0 = 0.00005;
    s0 = -0.3;
    pampams, pcov = curve_fit(lin_end,
                              yve, vaxe, p0 = [ue0, s0])
    up_edge = pampams[0]; slope = pampams[1];
    return up_edge, slope

def the_walls(data_folder, experiment, slice = 11):
    walls_etch_name = data_folder + "/run" + str(int(slice)) + \
        "/Im_channel_walls_mask_binary.png"
    real_walls_etch_name = data_folder + "/run" + str(int(slice)) + \
        "/Im_channel_walls_mask_binary_physical.png"
    walls_etch = cv.imread(walls_etch_name)
    real_walls_etch = cv.imread(real_walls_etch_name)
    walls_etch = cv.cvtColor(walls_etch.astype("uint8"), cv.COLOR_RGB2GRAY)
    real_walls_etch = cv.cvtColor(real_walls_etch.astype("uint8"), cv.COLOR_RGB2GRAY)

    leftwall = []; rightwall = []; width = [];
    for i in range(np.shape(walls_etch)[0]-1):
        for j in range(np.shape(walls_etch)[1]-1):
            if walls_etch[i,j] > 100 and walls_etch[i,j+1] < 100:
                leftwall.append(j)
            if walls_etch[i,j] < 100 and walls_etch[i,j+1] > 100:
                rightwall.append(j)
        width.append(rightwall[-1] - leftwall[-1])

    leftwall_real = []; rightwall_real = []; width_real = [];
    for i in range(np.shape(real_walls_etch)[0]-1):
        for j in range(np.shape(real_walls_etch)[1]-1):
            if real_walls_etch[i,j] > 100 and real_walls_etch[i,j+1] < 100:
                leftwall_real.append(j)
            if real_walls_etch[i,j] < 100 and real_walls_etch[i,j+1] > 100:
                rightwall_real.append(j)
        width_real.append(rightwall_real[-1] - leftwall_real[-1])

    width_mm = np.mean(width)*experiment.mmpx
    width_real_mm = np.mean(width_real)*experiment.mmpx

    max_deviation = experiment.mmpx*(np.max([abs(np.max(width) - np.mean(width)),
                                        abs(np.min(width) - np.mean(width))]))
    max_deviation_real = experiment.mmpx*(np.max([abs(np.max(width_real) - \
                                                      np.mean(width_real)),
                                        abs(np.min(width_real) - np.mean(width_real))]))
    return leftwall, rightwall, leftwall_real, rightwall_real, \
            width, width_real, width_mm, width_real_mm, \
            max_deviation, max_deviation_real

def image_parity(parent_dir, base_folder, subfolder):
    """
    Sort images in one subfolder of base_folder into even/odd folders.
    
    Args:
        parent_dir (str): Path to the directory containing base_folder.
        base_folder (str): Name of the main folder (e.g. 'preprocess').
        subfolder (str): Name of the subfolder inside base_folder to process.
    """
    # Absolute paths
    base_path = os.path.join(parent_dir, base_folder)
    subfolder_path = os.path.join(base_path, subfolder)

    # Output folders derived from base_folder name
    even_root = os.path.join(parent_dir, base_folder + "_evens")
    odd_root = os.path.join(parent_dir, base_folder + "_odds")

    # Create root even/odd directories if not exist
    os.makedirs(even_root, exist_ok=True)
    os.makedirs(odd_root, exist_ok=True)

    # Create the corresponding subfolders
    even_sub = os.path.join(even_root, subfolder)
    odd_sub = os.path.join(odd_root, subfolder)
    os.makedirs(even_sub, exist_ok=True)
    os.makedirs(odd_sub, exist_ok=True)

    # Regex to match filenames like "0XX.png"
    pattern = re.compile(r'^0(\d{2})\.png$')

    for fname in os.listdir(subfolder_path):
        match = pattern.match(fname)
        if not match:
            continue

        num = int(match.group(1))
        src_path = os.path.join(subfolder_path, fname)

        if num % 2 == 0:
            dest = os.path.join(even_sub, fname)
        else:
            dest = os.path.join(odd_sub, fname)

        shutil.copy2(src_path, dest)  # or use shutil.move

    print(f"Done sorting images in subfolder '{subfolder}'.")

# Example usage:
# sort_preprocess_images("/path/to/parent", "preprocess", "subfolder_name")

def ex_data(date, run):
    exd_name = "/more/Ex/piv/ex_data.txt"
    expr = Experiment([],[],[],[],[],[],[],[],[],[],
                      [],[],[],[],[],[])
    with open(exd_name) as read_obj:
        csv_dict_reader = DictReader(read_obj)
        csv_reader = csv.reader(read_obj, delimiter = ";")
        column_names = csv_dict_reader.fieldnames
        for row in csv_reader:
            if row[0] == date and int(row[1]) == run:
                expr.date = row[0]
                expr.run = int(row[1])
                expr.w = float(row[2])
                steps = row[3]
                expr.first_slice = int(row[4])
                expr.last_slice = int(row[5])
                expr.obj = row[6]
                expr.s_zoom = row[7]
                expr.f_dir = row[8]
                expr.x_start = int(row[9])
                expr.x_end = int(row[10])
                expr.f_rate = int(row[11])
                tauf = float(row[12])
                expr.fps = 1/tauf
                expr.sample = row[13]
    split_steps = steps[1:-1].split(', ')
    expr.steps = [int(s) for s in split_steps]
    if expr.obj == '4x':
        h10 = 1.64e-6
        expr.mmpx = 0.001*6.45/(0.7*4)
    elif expr.obj == '20x':
        h10 = 1.64e-6
        expr.mmpx = 0.001*6.45/(0.7*20)
    elif expr.obj == '20xA':
        h10 = 1.0e-6
        expr.mmpx = 0.001*6.45/(0.7*20)
    elif expr.obj == '63x':
        h10 = 1.64e-6
        expr.mmpx = 0.001*6.45/(0.7*63)
    y_axis = np.empty((expr.last_slice - expr.first_slice + 1, ))
    y_axis[0] = 0
    if expr.s_zoom == 'ud':
        h10 = -1*h10
    for i in range(1,np.size(y_axis)):
        y_axis[i] = y_axis[i-1] + 2*h10*expr.steps[i]
    expr.y_axis = y_axis + np.abs(np.min(y_axis))
    return expr

@dataclass
class Experiment:
    date: str
    run: int
    w: float
    first_slice: int
    last_slice: int
    obj: str
    s_zoom: str
    f_dir: str
    x_start: int
    x_end: int
    f_rate: int
    fps: float
    mmpx: float
    sample: str
    steps: list[int] = field(default_factory=list)
    y_axis: list[float] = field(default_factory=list)