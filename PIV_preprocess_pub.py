import os
import numpy as np
import cv2 as cv
from matplotlib import pyplot as plt
import multiprocessing
import scipy.signal as signl

# Function to read a pair of images from a multi-image .tif file
# Refer to readme for folder and file naming conventions used
def open_images_piv(image_fold, run_n, image_numb, id_this_run, reduce = False):
     im_name = image_fold + "/run" + str(run_n) + "." + id_this_run + ".0000" + \
            str(image_numb).zfill(2) + ".tif"
     img_fresh = cv.imreadmulti(im_name)

     if reduce:
          out_1 = img_fresh[1][0][:, reduce[0]:reduce[1]]
          out_2 = img_fresh[1][1][:, reduce[0]:reduce[1]]
     else:
          out_1, out_2 = img_fresh[1][0], img_fresh[1][1]

     return out_1, out_2

# Function to read a pair of images that have been written by histogram stretching
def open_images_stretched(image_fold, run_n, image_numb):
     im_name1 = image_fold + "/prestretch/slice_" + str(run_n).zfill(3) + \
            "/" + str(image_numb*2).zfill(3) + ".png"
     im_name2 = image_fold + "/prestretch/slice_" + str(run_n).zfill(3) + \
            "/" + str(image_numb*2 + 1).zfill(3) + ".png"
     
     return cv.imread(im_name1), cv.imread(im_name2)

# Conversion, optional save, optional display of an image
def image_preparation(image_given, show_it = False, save_it = False):
     image_out = np.full_like(image_given, 0, dtype=np.uint8)
     out_normal = cv.normalize(image_given, image_out, 0, 255, cv.NORM_MINMAX)
     img_out = cv.cvtColor(out_normal.astype("uint8"), cv.COLOR_GRAY2RGB);
     if save_it:
          cv.imwrite(save_it, img_out)
     if show_it:
          plt.figure(figsize=(1920/120, 1200/120), dpi=120)
          plt.axis('off')
          plt.imshow(img_out)

# Otains image file naming information
def name_runs(run_n, image_fold):
     these_files = os.listdir(image_fold)
     run_nums = []; run_ids = [];
     for i in range(0, np.size(these_files)-1):
          if these_files[i][4] == '.':
               run_nums.append(int(these_files[i][3]))
          else:
               run_nums.append(int(these_files[i][3:5]))
          run_ids.append(these_files[i][-19:-11])
     id_this_run = np.array(run_ids)[np.array(run_nums) == run_n][0]
     return id_this_run

# Displays histogram of an image, accenting sufficiently prominet peaks.
# Vertical lines illustrate points in histogram that would be used by
# histogram stretching, funct. "historic_cut"
def show_histogram(image, n_bins=52, cutoff_f=0.0025, dev_one = 2, wide = 1):
     hist, bin_edges = np.histogram(image, n_bins, [0,256])
     peaks, properties = signl.find_peaks(hist, width=wide, prominence=0.05e5)
     sufficiently_large = hist[hist>cutoff_f*np.sum(hist)]
     high_end = np.where(hist == sufficiently_large[-1])[0]
     low_end_1 = np.where(hist == sufficiently_large[0])[0]
     low_end_2 = peaks[-1] + \
            dev_one*(properties["right_ips"][-1] - properties["left_ips"][-1])
     plt.plot(hist)
     plt.plot(peaks, hist[peaks], "X")
     plt.hlines(y=properties["width_heights"], xmin=properties["left_ips"],
           xmax=properties["right_ips"], color = "C1")
     plt.vlines(x=low_end_1, ymin=0,
           ymax = hist[peaks[0]], color = "C1")
     plt.vlines(x=low_end_2, ymin=0,
           ymax = hist[peaks[-1]], color = "C1")
     if len(hist[peaks]) > 1:
          plt.vlines(x=high_end, ymin=0,
               ymax = hist[peaks[1]], color = "C1")
     else:
          plt.vlines(x=high_end, ymin=0,
               ymax = hist[peaks[0]], color = "C1")
     plt.show()

# Function for image gamma correction, takes image and gamma as arguments
def gamma_correction_table(image, gamma = 1.0):
     table = np.array([((i / 255.0) ** gamma) * 255
		for i in np.arange(0, 256)]).astype("uint8")
     im_gamma = cv.LUT(image.astype("uint8"), table)
     im_gamma[np.where(im_gamma < 0)] = 0
     return im_gamma
	
# Function for high-pass filter, takes image and kernel size as arguments
def highpass(img, kernel_size):
     blur = cv.GaussianBlur(img.astype("uint8"), (kernel_size, kernel_size), 0)
     im_blur = img - blur + 127
     return im_blur

# Function for low-pass filter, takes image and kernel size as arguments
def lowpass(img, k_size):
     kernl = np.ones([k_size,k_size])
     kernl = kernl/(3*sum(kernl))
     im_kern = cv.filter2D(img.astype("uint8"),-1,kernl)
     return im_kern

# Function to remove background, takes image and background as arguments
def remove_back(img, back):
     ib = img - back
     ib[np.where(ib < 0)] = 0
     return ib.astype("uint8")

# Function to identify min and max pixel intensity values for histogram stretching
def historic_cut(image_uncut, cutoff_f = 0.0025, n_bins = 52,
                  cut_method = 'sym_side', dev_one = 2, wide = 1):
     hist, bin_edges = np.histogram(image_uncut, n_bins, [0,256])
     sufficiently_large = hist[hist>cutoff_f*np.sum(hist)]
     high_end = bin_edges[np.where(hist == sufficiently_large[-1])][0]
     if cut_method == 'sym_side':
          low_end = bin_edges[np.where(hist == sufficiently_large[0])][0]
     elif cut_method == 'off_first_peak':
          hist, bin_edges = np.histogram(image_uncut, n_bins, [0,256])
          peaks, properties = signl.find_peaks(hist, width=wide, prominence=0.05e5)
          low_end = bin_edges[int(peaks[-1] + \
            dev_one*(properties["right_ips"][-1] - properties["left_ips"][-1]))]
     return low_end, high_end

# Translates the value of a pixel from a default interval of (0,255) to (lev1,lev2)
def stretch_dc(the_pixel, lev1, lev2):
     altered_pixel = np.min([np.max([the_pixel 
                                              - lev1,0])*(255/(lev2 - lev1)), 255])
     return altered_pixel

stretch_vector = np.vectorize(stretch_dc)

# ONE OF TWO PRIMARY PREPROCESSING METHODS
# Applies the "stretch_dc" function to an image, with an option to perform gamma corr.
# Pixel intensity values will be translated to interval (cut_low, cut_high)
def method_stretch(image, cut_low, cut_high, gamma = False):
     if gamma:
          image = gamma_correction_table(image, gamma)
     im_cut = stretch_vector(image, cut_low, cut_high)
     return im_cut

# Obtains background image for an image sequence (mean for the sequence)
# Returns two background images, intended for use with a sequence of image pairs
# Case 'piv' refers to original images
# Case 'stretched' refers to images for which "method_stretch" has been applied
def obtain_background(im_fold, run_id, specialk, id_this_run = False, all_range = 99):
     for i in range(0,all_range+1):
          if specialk == 'piv':
               im_1, im_2 = open_images_piv(im_fold, run_id, i, id_this_run)
          if specialk == 'stretched':
               im_1, im_2 = open_images_stretched(im_fold, run_id, i)
               im_1 = cv.cvtColor(im_1.astype("uint8"), cv.COLOR_RGB2GRAY)
               im_2 = cv.cvtColor(im_2.astype("uint8"), cv.COLOR_RGB2GRAY)
          if i == 0:
               im_mass_0 = np.empty([np.shape(im_1)[0], np.shape(im_1)[1], 100])
               im_mass_1 = np.empty_like(im_mass_0)
          im_mass_0[:,:,i] = im_1; im_mass_1[:,:,i] = im_2

     back_im_0 = np.ndarray.mean(im_mass_0, 2).astype(int)
     back_im_1 = np.ndarray.mean(im_mass_1, 2).astype(int)

     return back_im_0, back_im_1

# ONE OF TWO PRIMARY PREPROCESSING METHODS
# Applies high-pass and low-pass filters
# Optionally applies background removal and gamma correction
def method_two(im, high_kern = 15, low_kern = 3, brief = False, 
               back = False, im_background = False, extra_juice = False):

     if back:
          im_backed = remove_back(im, 0.9*im_background)
          im_low = lowpass(im_backed.astype("uint8"), low_kern)
     else:
          im_low = lowpass(im.astype("uint8"), low_kern)
     im_high = highpass(im_low, high_kern)
     im_high = gamma_correction_table(im_high, 2.5)

    # optional second gamma correction
     if extra_juice:
          im_high = gamma_correction_table(im_high, extra_juice)
          # extra_juice default = 1.5

     if brief:
          return im_high
     else:
          return im_high, im_low

# Function to apply histogram stretching, gamma correction & high/low pass to an image
# The intended use is function and parameter testing
def sequence_two(image_folder, run_n, im_n, method, step_analysis = False,
                  high_kern = 15, low_kern = 3, full_test = False, back = False,
                  extra_juice = False):
     if back:
          back_im_1, back_im_2 = obtain_background(image_folder,
                                               run_n, method)
          backcond = True
     else:
          back_im_1 = False; back_im_2 = False
          backcond = False
     
     im_1, im_2 = open_images_stretched(image_folder, run_n, im_n)
     im_1 = cv.cvtColor(im_1.astype("uint8"), cv.COLOR_RGB2GRAY)
     im_2 = cv.cvtColor(im_2.astype("uint8"), cv.COLOR_RGB2GRAY)

     im_high1, im_low1 = method_two(im_1, high_kern, low_kern, 
                                    back=backcond, im_background=back_im_1, 
                                    extra_juice=extra_juice)
     im_high2, im_low2 = method_two(im_2, high_kern, low_kern, 
                                    back=backcond, im_background=back_im_2,
                                    extra_juice=extra_juice)

     if full_test:
          try:
               os.makedirs(image_folder + "/preprocess/slice_" + str(run_n).zfill(3))
          except FileExistsError:
               pass
          im_out_name_1 = image_folder + "/preprocess/slice_" + str(run_n).zfill(3) + \
               "/" + str(im_n*2).zfill(3) + ".png"
          image_preparation(im_high1, show_it = False, save_it = im_out_name_1)

          im_out_name_2 = image_folder + "/preprocess/slice_" + str(run_n).zfill(3) + \
               "/" + str(im_n*2 + 1).zfill(3) + ".png"
          image_preparation(im_high2, show_it = False, save_it = im_out_name_2)
          
     else:
          image_preparation(im_high1, show_it = True, save_it = "test1.png")
          image_preparation(im_high2, show_it = True, save_it = "test2.png")

     if step_analysis:
          fig_s2,ax_s2 = plt.subplots(1,2,figsize=(12,10))
          ax_s2[0].imshow(im_high1,cmap=plt.cm.gray);
          ax_s2[1].imshow(im_high2,cmap=plt.cm.gray);
          fig_s3,ax_s3 = plt.subplots(1,2,figsize=(12,10))
          ax_s3[0].imshow(im_low1,cmap=plt.cm.gray);
          ax_s3[1].imshow(im_low2,cmap=plt.cm.gray);

# Function to apply histogram stretching, gamma correction & high/low pass to an image
# Writes the resulting image to file
def subsequence_two(im_n, image_folder, run_n, high_kern, low_kern, 
                    back_im_1, back_im_2, extra_juice = False, backcond = False):
     im_1, im_2 = open_images_stretched(image_folder, run_n, im_n)
     im_1 = cv.cvtColor(im_1.astype("uint8"), cv.COLOR_RGB2GRAY)
     im_2 = cv.cvtColor(im_2.astype("uint8"), cv.COLOR_RGB2GRAY)

     im_high1 = method_two(im_1, high_kern, low_kern, brief=True, 
                           back=backcond, im_background=back_im_1, 
                           extra_juice=extra_juice)
     im_high2 = method_two(im_2, high_kern, low_kern, brief=True, 
                           back=backcond, im_background=back_im_2,
                           extra_juice=extra_juice)

     im_out_name_1 = image_folder + "/preprocess/slice_" + str(run_n).zfill(3) + \
               "/" + str(im_n*2).zfill(3) + ".png"
     image_preparation(im_high1, show_it = False, save_it = im_out_name_1)

     im_out_name_2 = image_folder + "/preprocess/slice_" + str(run_n).zfill(3) + \
               "/" + str(im_n*2 + 1).zfill(3) + ".png"
     image_preparation(im_high2, show_it = False, save_it = im_out_name_2)

# Applies gamma correction (optional) and histogram stretching
# If out_fold (output folder) given: applies to image sequence, writes resutls to file
# Otherwise: applies to image pair, displays results (use for testing)
def sequence_stretch(image_fold, run_n, gamma = False,
                  image_range = 99, cutoff_f = 0.002, show_it = False, out_fold = False,
                    c_meth = 'sym_side', dev_one = 2, ends = False, back = False, wide=1):
     id_this_run = name_runs(run_n, image_fold)

     base_im_1, base_im_2 = open_images_piv(image_fold, run_n, 0, id_this_run, ends)

     if gamma:
          base_im_1 = gamma_correction_table(base_im_1, gamma)
          base_im_2 = gamma_correction_table(base_im_2, gamma)

     cut_low_1, cut_high_1 = historic_cut(base_im_1, cutoff_f,
                                           cut_method=c_meth, dev_one = dev_one,
                                           wide = wide)
     cut_low_2, cut_high_2 = historic_cut(base_im_2, cutoff_f,
                                           cut_method=c_meth, dev_one = dev_one,
                                           wide = wide)

     if out_fold:
          try:
               os.makedirs(out_fold + "/prestretch/slice_" + str(run_n).zfill(3))
          except FileExistsError:
               pass
          for image_numb in range(0,image_range):
               stretch_subsequence(image_numb, id_this_run, image_fold, 
                        cut_low_1, cut_high_1, cut_low_2, cut_high_2,
                         run_n, gamma, cutoff_f, out_fold, ends)
     else:
          im_1, im_2 = open_images_piv(image_fold, run_n, 0, id_this_run, ends)

          im_processed_1 = method_stretch(im_1, cut_low_1, cut_high_1, cutoff_f, gamma)
          im_processed_2 = method_stretch(im_2, cut_low_2, cut_high_2, cutoff_f, gamma)

          if show_it:
               image_preparation(im_processed_1, show_it)
               image_preparation(im_processed_2, show_it)
          else:
               fig_s1,ax_s1 = plt.subplots(1,2,figsize=(12,10))
               ax_s1[0].imshow(im_processed_1,cmap=plt.cm.gray);
               ax_s1[1].imshow(im_processed_2,cmap=plt.cm.gray);

# Subroutine for histogram stretching
# Writes resulting images to file
def stretch_subsequence(image_numb, id_this_run, image_fold, 
                        cut_low_1, cut_high_1, cut_low_2, cut_high_2,
                         run_n, gamma, cutoff_f, out_fold, ends = False):
     
     im_1, im_2 = open_images_piv(image_fold, run_n, image_numb, id_this_run, ends)

     im_processed_1 = method_stretch(im_1, cut_low_1, cut_high_1, cutoff_f, gamma)
     im_processed_2 = method_stretch(im_2, cut_low_2, cut_high_2, cutoff_f, gamma)

     im_out_name = out_fold + "/prestretch/slice_" + str(run_n).zfill(3) + \
               "/" + str(image_numb*2).zfill(3) + ".png"
     image_preparation(im_processed_1, show_it = False, save_it = im_out_name)

     im_out_name = out_fold + "/prestretch/slice_" + str(run_n).zfill(3) + \
               "/" + str(image_numb*2 + 1).zfill(3) + ".png"
     image_preparation(im_processed_2, show_it = False, save_it = im_out_name)

# Applies gamma correction (optional) and histogram stretching
# Expects TWO folders given:
# -folder where image files are stored (image_fold)
# -folder where subfolders with stretched images will be created (out_fold)
# Parallelized
def sequence_stretch_para(image_fold, run_n, core_n, gamma = False,
                  image_range = 99, cutoff_f = 0.002, out_fold = False,
                    c_meth = 'sym_side', dev_one = 2, ends = False, wide = 1):
     id_this_run = name_runs(run_n, image_fold)
     
     try:
          os.makedirs(out_fold + "/prestretch/slice_" + str(run_n).zfill(3))
     except FileExistsError:
          pass

     base_im_1, base_im_2 = open_images_piv(image_fold, run_n, 0, id_this_run, ends)

     if gamma:
          base_im_1 = gamma_correction_table(base_im_1, gamma)
          base_im_2 = gamma_correction_table(base_im_2, gamma)

     cut_low_1, cut_high_1 = historic_cut(base_im_1, cutoff_f,
                                           cut_method=c_meth, dev_one = dev_one,
                                           wide = wide)
     cut_low_2, cut_high_2 = historic_cut(base_im_2, cutoff_f,
                                           cut_method=c_meth, dev_one = dev_one,
                                           wide = wide)

     pool = multiprocessing.Pool(core_n)
     args = [(i , id_this_run, image_fold, cut_low_1, cut_high_1, cut_low_2, cut_high_2,
               run_n, gamma, cutoff_f, out_fold, ends) for i in range(0,image_range+1)]
     pool.starmap(stretch_subsequence, args)

# Applies histogram stretching, gamma correction & high/low pass to an image sequence
# image_fold is the folder containing histogram stretched image subfolders
# Parallelized
def sequence_many(image_fold, run_n, core_n, image_range = 99, all_range = 99,
                   high_kern = 5, low_kern = 2, back = False, extra_juice = False):
     if back:
          back_im_1, back_im_2 = obtain_background(image_fold, run_n,
                                               'stretched', all_range)
          backcond = True
     else:
          back_im_1 = False; back_im_2 = False; backcond = False
     
     try:
          os.makedirs(image_fold + "/preprocess/slice_" + str(run_n).zfill(3))
     except FileExistsError:
          pass

     args = [(im_n, image_fold, run_n,
                                 high_kern, low_kern, 
                                 back_im_1,
                                 back_im_2,
                                 extra_juice,
                                 backcond) for im_n in range(0, image_range+1)]
     
     pool = multiprocessing.Pool(core_n)
     pool.starmap(subsequence_two, args)