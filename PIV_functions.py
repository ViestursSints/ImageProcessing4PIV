from openpiv import tools, pyprocess, validation, windef
import multiprocessing
import numpy as np
import matplotlib.pyplot as plt
import pathlib
from IPython.display import clear_output

def settings_set(im_folder, slice_n, windows, overlaps, iter_n = 5,
                 std_thresh = 1, med_thresh = 3, normc = True):
    settings = windef.PIVSettings()
    #settings = windef.Settings()
    settings.filepath_images = im_folder + '/preprocess/slice_' + str(slice_n).zfill(3)
    settings.save_path = im_folder + '/PIV_output'
    settings.save_folder_suffix = 'slice_' + str(slice_n).zfill(3)

    settings.correlation_method = 'linear'
    settings.normalized_correlation = normc

    settings.num_iterations = iter_n

    settings.windowsizes = windows # (248, 164, 84, 64, 32)
    settings.overlap = overlaps # (132, 112, 32, 22, 16)

    settings.sig2noise_method = 'peak2peak'
    #settings.sig2noise_method = 'peak2mean'

    settings.validation_first_pass = False

    settings.min_max_u_disp = (-40, 40)
    settings.min_max_v_disp = (-40, 40)

    settings.std_threshold = std_thresh  # threshold of the std validation

    settings.median_threshold = med_thresh  # threshold of the median validation

    #settings.sig2noise_mask = 5
    settings.extract_sig2noise = False

    settings.do_sig2noise_validation = False

    settings.show_plot = False

    return settings

def settings_set_test(im_folder, slice_n, windows, overlaps, iter_n = 5,
                 std_thresh = 1, med_thresh = 3, normc = True):
    settings = windef.PIVSettings()
    #settings = windef.Settings()
    settings.filepath_images = im_folder + '/run' + str(slice_n) + '/2 Filtered Images'
    #settings.filepath_images = im_folder + '/run' + str(slice_n) + '/3 Particle Masks'
    settings.save_path = im_folder + '/PIV_output'
    settings.save_folder_suffix = 'slice_' + str(slice_n).zfill(3)

    settings.correlation_method = 'linear'
    settings.normalized_correlation = normc

    settings.num_iterations = iter_n

    settings.windowsizes = windows # (248, 164, 84, 64, 32)
    settings.overlap = overlaps # (132, 112, 32, 22, 16)

    settings.sig2noise_method = 'peak2peak'
    #settings.sig2noise_method = 'peak2mean'

    settings.validation_first_pass = False

    #settings.min_max_u_disp = (-40, 40)
    #settings.min_max_v_disp = (-40, 40)

    settings.std_threshold = std_thresh  # threshold of the std validation

    settings.median_threshold = med_thresh  # threshold of the median validation

    #settings.sig2noise_mask = 5
    #settings.extract_sig2noise = False

    #settings.do_sig2noise_validation = False

    settings.show_plot = False

    return settings

def do_piv(settings, n):
    im_name1 = str(n*2).zfill(3) + ".png"
    im_name2 = str(n*2 + 1).zfill(3) + ".png"
    #im_name1 = str(n*2 + 1).zfill(3) + ".png"
    #im_name2 = str(n*2 + 3).zfill(3) + ".png"

    settings.frame_pattern_a = im_name1
    settings.frame_pattern_b = im_name2
    windef.piv(settings)

def do_piv_skip(settings, n):
    im_name1 = str(n*2).zfill(3) + ".png"
    im_name2 = str(n*2 + 2).zfill(3) + ".png"
    #im_name1 = str(n*2 + 1).zfill(3) + ".png"
    #im_name2 = str(n*2 + 3).zfill(3) + ".png"

    settings.frame_pattern_a = im_name1
    settings.frame_pattern_b = im_name2
    windef.piv(settings)

def do_piv_mt(settings, n):
    im_name1 = "Frame_" + str(n).zfill(3) + ".tiff"
    im_name2 = "Frame_" + str(n + 1).zfill(3) + ".tiff"

    settings.frame_pattern_a = im_name1
    settings.frame_pattern_b = im_name2
    windef.piv(settings)

def mass_piv(settings, first_im = 1, im_range = 100, core_n = 10):
    args = [(settings, i) for i in range(first_im, im_range)]
    pool = multiprocessing.Pool(core_n)
    pool.starmap(do_piv, args)

def mass_piv_mt(settings, first_im = 1, im_range = 100, core_n = 10):
    args = [(settings, i) for i in range(first_im, im_range)]
    pool = multiprocessing.Pool(core_n)
    pool.starmap(do_piv_mt, args)

def piv_row(settings, first_im = 0, im_range = 200):
    settings.frame_pattern_a = '*.png'
    settings.frame_pattern_b = '(1+2),(3+4)'
    windef.piv(settings)

def piv_row_skip(settings, first_im = 0, im_range = 200):
    settings.frame_pattern_a = '*.png'
    settings.frame_pattern_b = '(1+3),(2+4)'
    windef.piv(settings)

def piv_row_parallel_skip(im_folder, slice, windows, overlaps,
                                  iter_n, std_thresh, med_thresh):
    settings = settings_set(im_folder, int(slice), windows, overlaps,
                                  iter_n, std_thresh = std_thresh, 
                                  med_thresh = med_thresh)
    settings.frame_pattern_a = '*.png'
    settings.frame_pattern_b = '(1+3),(2+4)'
    try:
        windef.piv(settings)
        clear_output(wait=True)
    except:
        print("There is trouble\n")

def piv_row_parallel(im_folder, slice, windows, overlaps,
                                  iter_n, std_thresh, med_thresh):
    settings = settings_set(im_folder, int(slice), windows, overlaps,
                                  iter_n, std_thresh = std_thresh, 
                                  med_thresh = med_thresh)
    settings.frame_pattern_a = '*.png'
    settings.frame_pattern_b = '(1+2),(3+4)'
    try:
        windef.piv(settings)
        clear_output(wait=True)
    except:
        print("There is trouble\n")

def piv_row_mte1(settings, first_im = 0, im_range = 100):
    settings.frame_pattern_a = '*.tiff'
    settings.frame_pattern_b = '(1+2),(3+4)'
    windef.piv(settings)

def piv_line(settings, first_im, im_range):
    for i in range(first_im, im_range):
        im_name1 = str(i*2).zfill(3) + ".png"
        im_name2 = str(i*2 + 1).zfill(3) + ".png"
        settings.frame_pattern_a = im_name1
        settings.frame_pattern_b = im_name2
        windef.piv(settings)

def displayer(slice_n, n, im_folder, window = 32):
    im_name1 = str(n*2).zfill(3) + ".png"
    fig_v, ax_v = plt.subplots(figsize=(8,8))
    tools.display_vector_field(
        pathlib.Path(im_folder +\
                     '/PIV_output/OpenPIV_results_' + str(int(window)) + '_slice_' + \
                      str(slice_n).zfill(3) + '/field_A0000.txt'),
        ax=ax_v, scaling_factor=10,
        scale=100, # scale defines here the arrow length
        width=0.003, # width is the thickness of the arrow
        on_img=True, # overlay on the image
        #image_name= thisplace + "\\" + im_name1
        image_name= im_folder + "/preprocess/slice_" + \
              str(slice_n).zfill(3) + '/' + im_name1
    );

def displayer_mt(slice_n, n, fn, im_folder, window = 32):
    im_name1 = "Frame_" + str(n) + ".tiff"
    fig_v, ax_v = plt.subplots(figsize=(8,8))
    tools.display_vector_field(
        pathlib.Path(im_folder +\
                     '/PIV_output/OpenPIV_results_' + str(int(window)) + '_slice_' + \
                      str(slice_n).zfill(3) + '/field_A00' + str(fn).zfill(2) + '.txt'),
        ax=ax_v, scaling_factor=10,
        scale=100, # scale defines here the arrow length
        width=0.003, # width is the thickness of the arrow
        on_img=True, # overlay on the image
        #image_name= thisplace + "\\" + im_name1
        image_name= im_folder + "/2 Filtered Images" + '/' + im_name1
        #image_name= im_folder + "/3 Particle Masks" + '/' + im_name1
        #image_name= im_folder + "/Gammad" + '/' + im_name1
    );

def displayer_mt2(slice_n, n, fn, im_folder, window = 32):
    im_name1 = "Frame_" + str(n).zfill(3) + ".tiff"
    fig_v, ax_v = plt.subplots(figsize=(8,8))
    tools.display_vector_field(
        pathlib.Path(im_folder +\
                     '/PIV_output/OpenPIV_results_' + str(int(window)) + '_slice_' + \
                      str(slice_n).zfill(3) + '/field_A00' + str(fn).zfill(2) + '.txt'),
        ax=ax_v, scaling_factor=1,
        scale=100, # scale defines here the arrow length
        width=0.003, # width is the thickness of the arrow
        on_img=True, # overlay on the image
        #image_name= thisplace + "\\" + im_name1
        #image_name= im_folder + "run" + 
        #    str(int(slice_n)) + "/2 Filtered Images" + '/' + im_name1
        image_name = im_folder + "run" + str(int(slice_n)) + \
                                 "/Im_tracerlines.jpg"
        #image_name= im_folder + "/3 Particle Masks" + '/' + im_name1
        #image_name= im_folder + "/Gammad" + '/' + im_name1
    );