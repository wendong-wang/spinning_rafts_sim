# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 16:08:51 2019

This is a test for committing on the local branch w541

@author: Wendong Wang, wwang@is.mpg.de
"""
import glob
import os
import shelve

import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import progressbar
# from scipy.integrate import RK45
from scipy.integrate import solve_ivp
from scipy.spatial import Voronoi as scipyVoronoi
# import scipy.io
from scipy.spatial import distance as scipy_distance


def draw_rafts_lh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the left-handed coordinate system of openCV
    positive x is pointing right
    positive y is pointing down
    """

    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1]), rafts_radii[raft_id],
                               circle_color, circle_thickness)

    return output_img


def draw_rafts_rh_coord(img_bgr, rafts_loc, rafts_radii, num_of_rafts):
    """
    draw circles in the right-handed coordinate system
    x pointing right
    y pointing up
    """
    circle_thickness = int(2)
    circle_color = (0, 0, 255)  # openCV: BGR

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    x_axis_start = (0, height - 10)
    x_axis_end = (width, height - 10)
    y_axis_start = (10, 0)
    y_axis_end = (10, height)
    output_img = cv.line(output_img, x_axis_start, x_axis_end, (0, 0, 0), 4)
    output_img = cv.line(output_img, y_axis_start, y_axis_end, (0, 0, 0), 4)

    for raft_id in np.arange(num_of_rafts):
        output_img = cv.circle(output_img, (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1]),
                               rafts_radii[raft_id], circle_color, circle_thickness)

    return output_img


def draw_b_field_in_rh_coord(img_bgr, b_orient):
    """
    draw the direction of B-field in right-handed xy coordinate
    """

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    line_length = 200
    line_start = (width // 2, height // 2)
    line_end = (int(width // 2 + np.cos(b_orient * np.pi / 180) * line_length),
                height - int(height // 2 + np.sin(b_orient * np.pi / 180) * line_length))
    output_img = cv.line(output_img, line_start, line_end, (0, 0, 0), 1)
    return output_img


def draw_raft_orientations_lh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft, as indicated by rafts_ori
    in left-handed coordiante system
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    int(rafts_loc[raft_id, 1] - np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[
                        raft_id]))  # note that the sign in front of the sine term is "-"
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_raft_orientations_rh_coord(img_bgr, rafts_loc, rafts_ori, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the dipole orientation of each raft, as indicated by rafts_ori
    in right-handed coordinate system
    """

    line_thickness = int(2)
    line_color = (255, 0, 0)

    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
        line_end = (int(rafts_loc[raft_id, 0] + np.cos(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[raft_id]),
                    height - int(rafts_loc[raft_id, 1] + np.sin(rafts_ori[raft_id] * np.pi / 180) * rafts_radii[
                        raft_id]))  # note that the sign in front of the sine term is "+"
        output_img = cv.line(output_img, line_start, line_end, line_color, line_thickness)

    return output_img


def draw_cap_peaks_lh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in left-handed coordinate system
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym

    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id, 0], rafts_loc[raft_id, 1])
            line_end = (int(
                rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap) * np.pi / 180) *
                rafts_radii[raft_id]),
                        int(rafts_loc[raft_id, 1] - np.sin(
                            (rafts_ori[raft_id] + cap_offset + capID * cap_gap) * np.pi / 180) * rafts_radii[
                                raft_id]))  # note that the sign in front of the sine term is "-"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_cap_peaks_rh_coord(img_bgr, rafts_loc, rafts_ori, raft_sym, cap_offset, rafts_radii, num_of_rafts):
    """
    draw lines to indicate the capillary peak positions
    in right-handed coordinate
    """

    line_thickness = int(2)
    line_color2 = (0, 255, 0)
    cap_gap = 360 / raft_sym
    #    cap_offset = 45 # the angle between the dipole direction and the first capillary peak

    output_img = img_bgr
    height, width, _ = img_bgr.shape
    for raft_id in np.arange(num_of_rafts):
        for capID in np.arange(raft_sym):
            line_start = (rafts_loc[raft_id, 0], height - rafts_loc[raft_id, 1])
            line_end = (int(
                rafts_loc[raft_id, 0] + np.cos((rafts_ori[raft_id] + cap_offset + capID * cap_gap) * np.pi / 180) *
                rafts_radii[raft_id]),
                        height - int(rafts_loc[raft_id, 1] + np.sin(
                            (rafts_ori[raft_id] + cap_offset + capID * cap_gap) * np.pi / 180) * rafts_radii[
                                         raft_id]))  # note that the sign in front of the sine term is "+"
            output_img = cv.line(output_img, line_start, line_end, line_color2, line_thickness)
    return output_img


def draw_raft_number(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1),
                                (rafts_loc[raft_id, 0] - text_size[0] // 2, rafts_loc[raft_id, 1] + text_size[1] // 2),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_raft_num_rh_coord(img_bgr, rafts_loc, num_of_rafts):
    """
    draw the raft number at the center of the rafts
    """

    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 2
    output_img = img_bgr
    height, width, _ = img_bgr.shape

    for raft_id in np.arange(num_of_rafts):
        text_size, _ = cv.getTextSize(str(raft_id + 1), font_face, font_scale, font_thickness)
        output_img = cv.putText(output_img, str(raft_id + 1), (rafts_loc[raft_id, 0] - text_size[0] // 2,
                                                               height - (rafts_loc[raft_id, 1] + text_size[1] // 2)),
                                font_face, font_scale, font_color, font_thickness, cv.LINE_AA)

    return output_img


def draw_frame_info(img_bgr, time_step_num, distance, orientation, b_field_direction, rel_orient):
    """
    draw information on the output frames
    """
    font_face = cv.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = (0, 0, 0)  # BGR
    font_thickness = 1
    output_img = img_bgr
    height, width, _ = img_bgr.shape
    text_size, _ = cv.getTextSize(str(time_step_num), font_face, font_scale, font_thickness)
    line_padding = 2
    left_padding = 20
    top_padding = 20
    output_img = cv.putText(output_img, 'time step: {}'.format(time_step_num), (left_padding, top_padding), font_face,
                            font_scale, font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'distance: {:03.2f}'.format(distance),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 1), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'orientation of raft 0: {:03.2f}'.format(orientation),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 2), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'B_field_direction: {:03.2f}'.format(b_field_direction),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 3), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    output_img = cv.putText(output_img, 'relative orientation phi_ji: {:03.2f}'.format(rel_orient),
                            (left_padding, top_padding + (text_size[1] + line_padding) * 4), font_face, font_scale,
                            font_color, font_thickness, cv.LINE_AA)
    # output_img = cv.putText(output_img, str('magnetic_dipole_force: {}'.format(magnetic_dipole_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 5), font_face,
    #                         font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('capillary_force: {}'.format(capillary_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 6),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('hydrodynamic_force: {}'.format(hydrodynamic_force)),
    #                         (10, 10 + (text_size[1] + line_padding) * 7),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('B-field_torque: {}'.format(B-field_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 8),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('mag_dipole_torque: {}'.format(mag_dipole_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 9),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)
    # output_img = cv.putText(output_img, str('cap_torque: {}'.format(cap_torque)),
    #                         (10, 10 + (text_size[1] + line_padding) * 10),
    #                         font_face, font_scale,font_color,font_thickness,cv.LINE_AA)

    return output_img


def draw_voronoi_rh_coord(img_bgr, rafts_loc):
    """
    draw Voronoi patterns
    """
    height, width, _ = img_bgr.shape
    points = rafts_loc
    points[:, 1] = height - points[:, 1]
    vor = scipyVoronoi(points)
    output_img = img_bgr
    # drawing Voronoi vertices
    vertex_size = int(3)
    vertex_color = (255, 0, 0)
    for x_pos, y_pos in zip(vor.vertices[:, 0], vor.vertices[:, 1]):
        output_img = cv.circle(output_img, (int(x_pos), int(y_pos)), vertex_size, vertex_color)

    # drawing Voronoi edges
    edge_color = (0, 255, 0)
    edge_thickness = int(2)
    for simplex in vor.ridge_vertices:
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            output_img = cv.line(output_img, (int(vor.vertices[simplex[0], 0]), int(vor.vertices[simplex[0], 1])),
                                 (int(vor.vertices[simplex[1], 0]), int(vor.vertices[simplex[1], 1])), edge_color,
                                 edge_thickness)

    center = points.mean(axis=0)
    for point_idx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.any(simplex < 0):
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = points[point_idx[1]] - points[point_idx[0]]  # tangent
            t = t / np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal
            midpoint = points[point_idx].mean(axis=0)
            far_point = vor.vertices[i] + np.sign(np.dot(midpoint - center, n)) * n * 200
            output_img = cv.line(output_img, (int(vor.vertices[i, 0]), int(vor.vertices[i, 1])),
                                 (int(far_point[0]), int(far_point[1])), edge_color, edge_thickness)
    return output_img


def map_angles_from0to360(angle_in_deg):
    """
    map the angleInDeg to the interval [0, 360)
        not in use anymore.
        % modulo operation just do the trick in one line!!
    """
    while angle_in_deg >= 360.0:
        angle_in_deg = angle_in_deg - 360.0
    while angle_in_deg < 0.0:
        angle_in_deg = angle_in_deg + 360.0

    return angle_in_deg


def fft_distances(sampling_rate, signal):
    """
    given sampling rate and signal, output frequency vector and one-sided power spectrum
    sampling_rate: unit Hz
    signal: numpy array
    """
    #    sampling_interval = 1/sampling_rate # unit s
    #    times = np.linspace(0,sampling_length*sampling_interval, sampling_length)
    sampling_length = len(signal)  # total number of frames
    fft = np.fft.fft(signal)
    p2 = np.abs(fft / sampling_length)
    p1 = p2[0:int(sampling_length / 2) + 1]
    p1[1:-1] = 2 * p1[1:-1]  # one-sided power spectrum
    frequencies = sampling_rate / sampling_length * np.arange(0, int(sampling_length / 2) + 1)

    return frequencies, p1


def adjust_phases(phases_input):
    """
    adjust the phases to get rid of the jump of 360 when it crosses from -180 to 180, or the reverse
    adjust single point anormaly.
    """
    phase_diff_threshold = 200

    phases_diff = np.diff(phases_input)

    index_neg = phases_diff < -phase_diff_threshold
    index_pos = phases_diff > phase_diff_threshold

    insertion_indices_neg = np.nonzero(index_neg)
    insertion_indices_pos = np.nonzero(index_pos)

    phase_diff_corrected = phases_diff.copy()
    phase_diff_corrected[insertion_indices_neg[0]] += 360
    phase_diff_corrected[insertion_indices_pos[0]] -= 360

    phases_corrected = phases_input.copy()
    phases_corrected[1:] = phase_diff_corrected[:]
    phases_adjusted = np.cumsum(phases_corrected)

    return phases_adjusted


# %% load capillary force and torque
rootFolderNameFromWindows = r'D:\\SimulationFolder\spinningRaftsSimulationCode'

os.chdir(rootFolderNameFromWindows)

os.chdir('2019-05-13_capillaryForceCalculations-sym6')  # this is for sym4 rafts

shelveName = 'capillaryForceAndTorque_sym6'
shelveDataFileName = shelveName + '.dat'
listOfVariablesToLoad = ['eeDistanceCombined', 'forceCombinedDistancesAsRowsAll360',
                         'torqueCombinedDistancesAsRowsAll360']

if not os.path.isfile(shelveDataFileName):
    print('the capillary data file is missing')

tempShelf = shelve.open(shelveName)
capillaryEEDistances = tempShelf['eeDistanceCombined']  # unit: m
capillaryForcesDistancesAsRowsLoaded = tempShelf['forceCombinedDistancesAsRowsAll360']  # unit: N
capillaryTorquesDistancesAsRowsLoaded = tempShelf['torqueCombinedDistancesAsRowsAll360']  # unit: N.m

os.chdir('..')

# further data treatment on capillary force profile
# insert the force and torque at eeDistance = 1um as the value for eedistance = 0um.
capillaryEEDistances = np.insert(capillaryEEDistances, 0, 0)
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRowsLoaded[:1, :], capillaryForcesDistancesAsRowsLoaded), axis=0)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRowsLoaded[:1, :], capillaryTorquesDistancesAsRowsLoaded), axis=0)

# add angle=360, the same as angle = 0
capillaryForcesDistancesAsRows = np.concatenate(
    (capillaryForcesDistancesAsRows, capillaryForcesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)
capillaryTorquesDistancesAsRows = np.concatenate(
    (capillaryTorquesDistancesAsRows, capillaryTorquesDistancesAsRows[:, 0].reshape(1001, 1)), axis=1)

# correct for the negative sign of the torque
capillaryTorquesDistancesAsRows = - capillaryTorquesDistancesAsRows

# some extra treatment for the force matrix
# note the sharp transition at the peak-peak position (45 deg): only 1 deg difference,
# the force changes from attraction to repulsion. consider replacing values at eeDistance = 0, 1, 2,
# with values at eeDistance = 5um.
nearEdgeSmoothingThres = 1  # unit: micron; if 1, then it is equivalent to no smoothing.
for distanceToEdge in np.arange(nearEdgeSmoothingThres):
    capillaryForcesDistancesAsRows[distanceToEdge, :] = capillaryForcesDistancesAsRows[nearEdgeSmoothingThres, :]
    capillaryTorquesDistancesAsRows[distanceToEdge, :] = capillaryTorquesDistancesAsRows[nearEdgeSmoothingThres, :]

# select a cut-off distance below which all the attractive force (negative-valued) becomes zero,
# due to raft wall-wall repulsion
capAttractionZeroCutoff = 0
mask = np.concatenate((capillaryForcesDistancesAsRows[:capAttractionZeroCutoff, :] < 0,
                       np.zeros((capillaryForcesDistancesAsRows.shape[0] - capAttractionZeroCutoff,
                                 capillaryForcesDistancesAsRows.shape[1]), dtype=int)),
                      axis=0)
capillaryForcesDistancesAsRows[mask.nonzero()] = 0

# set capillary force = 0 at 0 distance
# capillaryForcesDistancesAsRows[0,:] = 0

# realign the first peak-peak direction with an angle = capillaryPeakOffset from the x-axis.
capillaryPeakOffset = 0
capillaryForcesDistancesAsRows = np.roll(capillaryForcesDistancesAsRows, capillaryPeakOffset,
                                         axis=1)  # 45 is due to original data
capillaryTorquesDistancesAsRows = np.roll(capillaryTorquesDistancesAsRows, capillaryPeakOffset, axis=1)

capillaryForceAngleAveraged = capillaryForcesDistancesAsRows[1:, :-1].mean(axis=1)  # starting from 1 um to 1000 um
capillaryForceMaxRepulsion = capillaryForcesDistancesAsRows[1:, :-1].max(axis=1)
capillaryForceMaxRepulsionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmax(axis=1)
capillaryForceMaxAttraction = capillaryForcesDistancesAsRows[1:, :-1].min(axis=1)
capillaryForceMaxAttractionIndex = capillaryForcesDistancesAsRows[1:, :-1].argmin(axis=1)

# %% magnetic force and torque calculation:
miu0 = 4 * np.pi * 1e-7  # unit: N/A**2, or T.m/A, or H/m

# from the data 2018-09-28, 1st increase:
# (1.4e-8 A.m**2 for 14mT), (1.2e-8 A.m**2 for 10mT), (0.96e-8 A.m**2 for 5mT), (0.78e-8 A.m**2 for 1mT)
# from the data 2018-09-28, 2nd increase:
# (1.7e-8 A.m**2 for 14mT), (1.5e-8 A.m**2 for 10mT), (1.2e-8 A.m**2 for 5mT), (0.97e-8 A.m**2 for 1mT)
magneticMomentOfOneRaft = 1e-8  # unit: A.m**2

orientationAngles = np.arange(0, 361)  # unit: degree;
orientationAnglesInRad = np.radians(orientationAngles)

magneticDipoleEEDistances = np.arange(0, 10001) / 1e6  # unit: m

radiusOfRaft = 1.5e-4  # unit: m

magneticDipoleCCDistances = magneticDipoleEEDistances + radiusOfRaft * 2  # unit: m

# magDpEnergy = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: J
magDpForceOnAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpForceOffAxis = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N
magDpTorque = np.zeros((len(magneticDipoleEEDistances), len(orientationAngles)))  # unit: N.m

for index, d in enumerate(magneticDipoleCCDistances):
    # magDpEnergy[index, :] = \
    #     miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 3)
    magDpForceOnAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (1 - 3 * (np.cos(orientationAnglesInRad) ** 2)) / (4 * np.pi * d ** 4)
    magDpForceOffAxis[index, :] = \
        3 * miu0 * magneticMomentOfOneRaft ** 2 * (2 * np.cos(orientationAnglesInRad) *
                                                   np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 4)
    magDpTorque[index, :] = \
        miu0 * magneticMomentOfOneRaft ** 2 * (3 * np.cos(orientationAnglesInRad) *
                                               np.sin(orientationAnglesInRad)) / (4 * np.pi * d ** 3)

# magnetic force at 1um(attractionZeroCutoff) should have no attraction, due to wall-wall repulsion.
# Treat it similarly as capillary cutoff
attractionZeroCutoff = 0  # unit: micron
mask = np.concatenate((magDpForceOnAxis[:attractionZeroCutoff, :] < 0,
                       np.zeros((magDpForceOnAxis.shape[0] - attractionZeroCutoff, magDpForceOnAxis.shape[1]),
                                dtype=int)), axis=0)
magDpForceOnAxis[mask.nonzero()] = 0

magDpMaxRepulsion = magDpForceOnAxis.max(axis=1)
magDpForceAngleAverage = magDpForceOnAxis[:, :-1].mean(axis=1)

# set on-axis magnetic force = 0 at 0 distance
# magDpForceOnAxis[0,:] = 0

# %% lubrication equation coefficients:
RforCoeff = 150.0  # unit: micron
stepSizeForDist = 0.1
lubCoeffScaleFactor = 1 / stepSizeForDist
eeDistancesForCoeff = np.arange(0, 15 + stepSizeForDist, stepSizeForDist, dtype='double')  # unit: micron

eeDistancesForCoeff[0] = 1e-10  # unit: micron

x = eeDistancesForCoeff / RforCoeff  # unit: 1

lubA = x * (-0.285524 * x + 0.095493 * x * np.log(x) + 0.106103) / RforCoeff  # unit: 1/um

lubB = ((0.0212764 * (- np.log(x)) + 0.157378) * (- np.log(x)) + 0.269886) / (
        RforCoeff * (- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549)  # unit: 1/um

# lubC = ( (-0.0212758 * (- np.log(x)) - 0.089656) * (- np.log(x)) + 0.0480911) / (RforCoeff**2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^2

# lubD = (0.0579125 * (- np.log(x)) + 0.0780201) / (RforCoeff**2 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^2

lubG = ((0.0212758 * (- np.log(x)) + 0.181089) * (- np.log(x)) + 0.381213) / (
        RforCoeff ** 3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549))  # unit: 1/um^3

lubC = - RforCoeff * lubG

# lubH = (0.265258 * (- np.log(x)) + 0.357355) / (RforCoeff**3 * ((- np.log(x)) * ((- np.log(x)) + 6.0425) + 6.32549) ) # unit: 1/um^3

# lubCoeffCombined = np.column_stack((lubA,lubB,lubC,lubD,lubG,lubH))
# %% simulation of the pairwise data first,
# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit
#
## check the dipole orientation and capillary orientation
eeDistanceForPlotting = 70
fig, ax = plt.subplots(ncols=2, nrows=1)
ax[0].plot(capillaryForcesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
           label='capillary force')  # 0 deg is the peak-peak alignment - attraction.
ax[0].plot(magDpForceOnAxis[eeDistanceForPlotting, :], 'o-',
           label='magnetic force')  # 0 deg is the dipole-dipole attraction
ax[0].set_xlabel('angle')
ax[0].set_ylabel('force (N)')
ax[0].legend()
ax[0].set_title('force at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
                                                                                         capillaryPeakOffset))
ax[1].plot(capillaryTorquesDistancesAsRows[eeDistanceForPlotting, :], 'o-',
           label='capillary torque')  # 0 deg is the peak-peak alignment - attraction.
ax[1].plot(magDpTorque[eeDistanceForPlotting, :], 'o-',
           label='magnetic torque')  # 0 deg is the dipole-dipole attraction
ax[1].set_xlabel('angle')
ax[1].set_ylabel('torque (N.m)')
ax[1].legend()
ax[1].set_title('torque at eeDistance = {}um, capillary peak offset angle = {}deg'.format(eeDistanceForPlotting,
                                                                                          capillaryPeakOffset))

# plot the various forces and look for the transition rps
# densityOfWater = 1e-15 # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
# raftRadius = 1.5e2 # unit: micron
# magneticFieldRotationRPS = 22
# omegaBField = magneticFieldRotationRPS * 2 * np.pi
# hydrodynamicRepulsion = densityOfWater * omegaBField**2 * raftRadius**7 * 1e-6/ np.arange(raftRadius * 2 + 1, raftRadius * 2 + 1002)**3  #unit: N
# sumOfAllForces = capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001] + hydrodynamicRepulsion
# fig, ax = plt.subplots(ncols = 1, nrows = 1)
# ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1), label = 'angle-averaged capillary force')
# ax.plot(magDpForceOnAxis.mean(axis = 1)[:1000], label = 'angle-averaged magnetic force')
# ax.plot(hydrodynamicRepulsion, label = 'hydrodynamic repulsion')
##ax.plot(capillaryForcesDistancesAsRows.mean(axis = 1) + magDpForceOnAxis.mean(axis = 1)[:1001], label = 'angle-avaraged sum of magnetic and capillary force')
# ax.set_xlabel('edge-edge distance (um)')
# ax.set_ylabel('Force (N)')
# ax.set_title('spin speed {} rps'.format(magneticFieldRotationRPS))
# ax.plot(sumOfAllForces, label = 'sum of angle-averaged magnetic and capillary forces and hydrodynamic force ')
# ax.legend()

os.chdir(r'D:\SimulationFolder')

listOfVariablesToSave = ['numOfRafts', 'magneticFieldStrength', 'magneticFieldRotationRPS', 'omegaBField',
                         'timeStepSize', 'numOfTimeSteps',
                         'timeTotal', 'outputImageSeq', 'outputVideo', 'outputFrameRate', 'intervalBetweenFrames',
                         'raftLocations', 'raftOrientations', 'raftRadii', 'raftRotationSpeedsInRad',
                         'raftRelativeOrientationInDeg',
                         #                         'velocityTorqueCouplingTerm', 'magDipoleForceOffAxisTerm','magDipoleForceOnAxisTerm', 'capillaryForceTerm',
                         #                         'hydrodynamicForceTerm', 'stochasticTerm', 'forceCurvatureTerm', 'wallRepulsionTerm',
                         #                         'magneticFieldTorqueTerm', 'magneticDipoleTorqueTerm', 'capillaryTorqueTerm',
                         'currentStepNum', 'currentFrameBGR']
# small number threshold
# eps = 1e-13

# constant of proportionalities
cm = 1  # coefficient for the magnetic force term
cc = 1  # coefficient for the capillary force term
ch = 1  # coefficient for the hydrodynamic force term
tb = 1  # coefficient for the magnetic field torque term
tm = 1  # coefficient for the magnetic dipole-dipole torque term
tc = 1  # coefficient for the capillary torque term
forceDueToCurvature = 0  # unit: N
wallRepulsionForce = 1e-7  # unit: N
# elasticWallThickness = 5 # unit: micron


arenaSize = 2e3  # unit: micron 2e3, 5e3,
R = raftRadius = 1.5e2  # unit: micron
centerOfArena = np.array([arenaSize / 2, arenaSize / 2])

# all calculations are done in SI numbers, and only in drawing are the variables converted to pixel unit
canvasSizeInPixel = int(1000)  # unit: pixel
scaleBar = arenaSize / canvasSizeInPixel  # unit: micron/pixel

densityOfWater = 1e-15  # unit conversion: 1000 kg/m^3 = 1e-15 kg/um^3
miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
piMiuR = np.pi * miu * raftRadius  # unit: N.s/um

numOfRafts = 2
magneticFieldStrength = 10e-3  # 14e-3 #10e-3 # unit: T
initialPositionMethod = 2  # 1 -random positions, 2 - fixed initial position, 3 - starting positions are the last positions of the previous spin speeds
ccSeparationStarting = 400  # unit: micron
initialOrientation = 0  # unit: deg
lastPositionOfPreviousSpinSpeeds = np.zeros((numOfRafts, 2))
lastOmegaOfPreviousSpinSpeeds = np.zeros((numOfRafts))
firstSpinSpeedFlag = 1

timeStepSize = 1e-3  # unit: s
numOfTimeSteps = 50000
timeTotal = timeStepSize * numOfTimeSteps

lubEqThreshold = 15  # unit micron, if the eeDistance is below this value, the torque velocity coupling term changes to rigid body rotation
stdOfFluctuationTerm = 0.00
stdOfTorqueNoise = 0  # 1e-12 # unit: N.m

outputImageSeq = 0
outputVideo = 1
outputFrameRate = 10.0
intervalBetweenFrames = int(10)  # unit: steps
blankFrameBGR = np.ones((canvasSizeInPixel, canvasSizeInPixel, 3), dtype='int') * 255

solverMethod = 'RK45'  # RK45, RK23, Radau, BDF, LSODA


def Fun_drdt_dalphadt(t, raft_loc_orient):
    '''
    Two sets of ordinary differential equations that define dr/dt and dalpha/dt above and below the threshold value
    for the application of lubrication equations
    '''
    #    raft_loc_orient = raftLocationsOrientations
    raft_loc = raft_loc_orient[0: numOfRafts * 2].reshape(numOfRafts, 2)  # in um
    raft_orient = raft_loc_orient[numOfRafts * 2: numOfRafts * 3]  # in deg

    drdt = np.zeros((numOfRafts, 2))  # unit: um
    raft_spin_speeds_inRads = np.zeros(numOfRafts)  # in rad
    dalphadt = np.zeros(numOfRafts)  # unit: deg

    mag_Dipole_Force_OnAxis_Term = np.zeros((numOfRafts, 2))
    capillary_Force_Term = np.zeros((numOfRafts, 2))
    hydrodynamic_Force_Term = np.zeros((numOfRafts, 2))
    mag_Dipole_Force_OffAxis_Term = np.zeros((numOfRafts, 2))
    velocity_Torque_Coupling_Term = np.zeros((numOfRafts, 2))
    velocity_Mag_Fd_Torque_Term = np.zeros((numOfRafts, 2))
    wall_Repulsion_Term = np.zeros((numOfRafts, 2))
    stochastic_Force_Term = np.zeros((numOfRafts, 2))
    force_Curvature_Term = np.zeros((numOfRafts, 2))

    magnetic_Field_Torque_Term = np.zeros(numOfRafts)
    magnetic_Dipole_Torque_Term = np.zeros(numOfRafts)
    capillary_Torque_Term = np.zeros(numOfRafts)
    stochastic_Torque_Term = np.zeros(numOfRafts)

    # stochastic torque term
    #    stochastic_Torque = omegaBField * np.random.normal(0, stdOfTorqueNoise, 1) # unit: N.m, assuming omegaBField is unitless
    #    stochastic_Torque_Term = np.ones(numOfRafts) * stochastic_Torque * 1e6 /(8*piMiuR*R**2) # unit: 1/s assuming omegaBField is unitless.

    # loop for torques and calculate raft_spin_speeds_inRads
    for raftID in np.arange(numOfRafts):
        # raftID = 0
        ri = raft_loc[raftID, :]  # unit: micron

        # magnetic field torque:
        magnetic_Field_Torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(
            np.deg2rad(magneticFieldDirection - raft_orient[raftID]))  # unit: N.m
        magnetic_Field_Torque_Term[raftID] = tb * magnetic_Field_Torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        rji_eeDist_smallest = R;  # initialize

        for neighborID in np.arange(numOfRafts):
            if neighborID == raftID:
                continue
            rj = raft_loc[neighborID, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_Norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_eeDist = rji_Norm - 2 * R  # unit: micron
            rji_Unitized = rji / rji_Norm  # unit: micron
            rji_Unitized_CrossZ = np.asarray((rji_Unitized[1], -rji_Unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[
                raftID]) % 360  # unit: deg; assuming both rafts's orientations are the same

            #            print('{}, {}'.format(int(phi_ji), (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[raftID])))
            #             torque terms:
            if rji_eeDist < lubEqThreshold and rji_eeDist < rji_eeDist_smallest:
                rji_eeDist_smallest = rji_eeDist
                if rji_eeDist_smallest >= 0:
                    magnetic_Field_Torque_Term[raftID] = lubG[int(
                        rji_eeDist_smallest * lubCoeffScaleFactor)] * magnetic_Field_Torque * 1e6 / miu  # unit: 1/s
                elif rji_eeDist_smallest < 0:
                    magnetic_Field_Torque_Term[raftID] = lubG[0] * magnetic_Field_Torque * 1e6 / miu  # unit: 1/s

            if rji_eeDist < 10000 and rji_eeDist >= 0:
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm * magDpTorque[
                    int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / (8 * piMiuR * R ** 2)
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm * lubG[
                    int(rji_eeDist * lubCoeffScaleFactor)] * magDpTorque[int(rji_eeDist + 0.5), int(
                    phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s
            elif rji_eeDist < 0:
                magnetic_Dipole_Torque_Term[raftID] = magnetic_Dipole_Torque_Term[raftID] + tm * lubG[0] * magDpTorque[
                    0, int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s

            if rji_eeDist < 1000 and rji_eeDist >= lubEqThreshold:
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * capillaryTorquesDistancesAsRows[
                    int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * lubG[
                    int(rji_eeDist * lubCoeffScaleFactor)] * capillaryTorquesDistancesAsRows[
                                                    int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s
            elif rji_eeDist < 0:
                capillary_Torque_Term[raftID] = capillary_Torque_Term[raftID] + tc * lubG[0] * \
                                                capillaryTorquesDistancesAsRows[
                                                    0, int(phi_ji + 0.5)] * 1e6 / miu  # unit: 1/s

            # debug use:
        #            raftRelativeOrientationInDeg[neighborID, raftID, currentStepNum] = phi_ji

        # debug use
        #        capillaryTorqueTerm[raftID, currentStepNum] = capillary_Torque_Term[raftID]

        raft_spin_speeds_inRads[raftID] = stochastic_Torque_Term[raftID] + magnetic_Field_Torque_Term[raftID] + \
                                          magnetic_Dipole_Torque_Term[raftID] + capillary_Torque_Term[raftID]

    # loop for forces
    for raftID in np.arange(numOfRafts):
        # raftID = 0
        ri = raft_loc[raftID, :]  # unit: micron

        # force cuvature term
        if forceDueToCurvature != 0:
            ri_center = centerOfArena - ri
            #            ri_center_Norm = np.sqrt(ri_center[0]**2 + ri_center[1]**2)
            #            ri_center_Unitized = ri_center / ri_center_Norm
            force_Curvature_Term[raftID, :] = forceDueToCurvature / (6 * piMiuR) * ri_center / (arenaSize / 2)

        # magnetic field torque:
        magnetic_Field_Torque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(
            np.deg2rad(magneticFieldDirection - raft_orient[raftID]))  # unit: N.m
        magnetic_Field_Torque_Term[raftID] = tb * magnetic_Field_Torque * 1e6 / (8 * piMiuR * R ** 2)  # unit: 1/s

        for neighborID in np.arange(numOfRafts):
            if neighborID == raftID:
                continue
            rj = raft_loc[neighborID, :]  # unit: micron
            rji = ri - rj  # unit: micron
            rji_Norm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
            rji_eeDist = rji_Norm - 2 * R  # unit: micron
            rji_Unitized = rji / rji_Norm  # unit: micron
            rji_Unitized_CrossZ = np.asarray((rji_Unitized[1], -rji_Unitized[0]))
            phi_ji = (np.arctan2(rji[1], rji[0]) * 180 / np.pi - raft_orient[
                raftID]) % 360  # unit: deg; assuming both rafts's orientations are the same, modulo operation remember!
            if phi_ji == 360: phi_ji = 0
            #            raft_Relative_Orientation_InDeg[neighborID, raftID] = phi_ji

            # force terms:
            omegaj = raft_spin_speeds_inRads[
                neighborID]  # need to come back and see how to deal with this. maybe you need to define it as a global variable.

            if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold:
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm * \
                                                          magDpForceOnAxis[int(rji_eeDist + 0.5), int(
                                                              phi_ji + 0.5)] * rji_Unitized / (6 * piMiuR)  # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm * lubA[
                    int(rji_eeDist * lubCoeffScaleFactor)] * magDpForceOnAxis[int(rji_eeDist + 0.5), int(
                    phi_ji + 0.5)] * rji_Unitized / miu  # unit: um/s
            elif rji_eeDist < 0:
                mag_Dipole_Force_OnAxis_Term[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + cm * lubA[0] * \
                                                          magDpForceOnAxis[
                                                              0, int(phi_ji + 0.5)] * rji_Unitized / miu  # unit: um/s

            if rji_eeDist < 1000 and rji_eeDist >= lubEqThreshold:
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc * capillaryForcesDistancesAsRows[
                    int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized / (6 * piMiuR)  # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc * lubA[
                    int(rji_eeDist * lubCoeffScaleFactor)] * capillaryForcesDistancesAsRows[int(rji_eeDist + 0.5), int(
                    phi_ji + 0.5)] * rji_Unitized / miu  # unit: um/s
            elif rji_eeDist < 0:
                capillary_Force_Term[raftID, :] = capillary_Force_Term[raftID, :] + cc * lubA[0] * \
                                                  capillaryForcesDistancesAsRows[
                                                      0, int(phi_ji + 0.5)] * rji_Unitized / miu  # unit: um/s

            if rji_eeDist >= lubEqThreshold:
                hydrodynamic_Force_Term[raftID, :] = hydrodynamic_Force_Term[raftID,
                                                     :] + ch * 1e-6 * densityOfWater * omegaj ** 2 * R ** 7 * rji / rji_Norm ** 4 / (
                                                             6 * piMiuR)  # unit: um/s; 1e-6 is used to convert the implicit m to um in Newton in miu
            elif rji_eeDist < lubEqThreshold and rji_eeDist > 0:
                hydrodynamic_Force_Term[raftID, :] = hydrodynamic_Force_Term[raftID, :] + ch * lubA[
                    int(rji_eeDist * lubCoeffScaleFactor)] * (
                                                             1e-6 * densityOfWater * omegaj ** 2 * R ** 7 / rji_Norm ** 3) * rji_Unitized / miu  # unit: um/s

            if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold:
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + magDpForceOffAxis[
                    int(rji_eeDist + 0.5), int(phi_ji + 0.5)] * rji_Unitized_CrossZ / (6 * piMiuR)
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + lubB[
                    int(rji_eeDist * lubCoeffScaleFactor)] * magDpForceOffAxis[int(rji_eeDist + 0.5), int(
                    phi_ji + 0.5)] * rji_Unitized_CrossZ / miu  # unit: um/s
            elif rji_eeDist < 0:
                mag_Dipole_Force_OffAxis_Term[raftID, :] = mag_Dipole_Force_OffAxis_Term[raftID, :] + lubB[0] * \
                                                           magDpForceOffAxis[0, int(
                                                               phi_ji + 0.5)] * rji_Unitized_CrossZ / miu  # unit: um/s

            if rji_eeDist >= lubEqThreshold:
                velocity_Torque_Coupling_Term[raftID, :] = velocity_Torque_Coupling_Term[raftID,
                                                           :] - R ** 3 * omegaj * rji_Unitized_CrossZ / (
                                                                   rji_Norm ** 2)  # unit: um/s
            elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
                velocity_Mag_Fd_Torque_Term[raftID, :] = velocity_Mag_Fd_Torque_Term[raftID, :] + lubC[int(
                    rji_eeDist * lubCoeffScaleFactor)] * magnetic_Field_Torque * 1e6 * rji_Unitized_CrossZ / miu  # unit: um/s
            elif rji_eeDist < 0:
                velocity_Mag_Fd_Torque_Term[raftID, :] = velocity_Mag_Fd_Torque_Term[raftID, :] + lubC[
                    0] * magnetic_Field_Torque * 1e6 * rji_Unitized_CrossZ / miu  # unit: um/s

            #                if rji_eeDist >= lubEqThreshold and currentStepNum > 1:
            #                    prev_drdt = (raftLocations[raftID,currentStepNum,:] -  raftLocations[raftID,currentStepNum-1,:]) / timeStepSize
            #                    stochastic_Force_Term[raftID, currentStepNum,:] =  stochastic_Force_Term[raftID, currentStepNum,:] + np.sqrt(prev_drdt[0]**2 + prev_drdt[1]**2) * np.random.normal(0, stdOfFluctuationTerm, 1) * rjiUnitized

            if rji_eeDist < 0:
                wall_Repulsion_Term[raftID, :] = wall_Repulsion_Term[raftID, :] + wallRepulsionForce / (6 * piMiuR) * (
                        -rji_eeDist / R) * rji_Unitized

        # update drdr and dalphadt
        drdt[raftID, :] = mag_Dipole_Force_OnAxis_Term[raftID, :] + capillary_Force_Term[raftID,
                                                                    :] + hydrodynamic_Force_Term[raftID, :] \
                          + mag_Dipole_Force_OffAxis_Term[raftID, :] + velocity_Torque_Coupling_Term[raftID,
                                                                       :] + velocity_Mag_Fd_Torque_Term[raftID, :] \
                          + stochastic_Force_Term[raftID, :] + wall_Repulsion_Term[raftID, :] + force_Curvature_Term[
                                                                                                raftID, :]

    dalphadt = raft_spin_speeds_inRads / np.pi * 180  # in deg

    drdt_dalphadt = np.concatenate((drdt.flatten(), dalphadt))

    return drdt_dalphadt


# for stdOfFluctuationTerm in np.arange(0.01,0.11,0.04):
for magneticFieldRotationRPS in np.arange(-1, -30,
                                          -1):  # negative means clockwise in Rhino coordinate, and positive means counter-clockwise in Rhino coordinate
    #    magneticFieldRotationRPS = -10 # unit: rps (rounds per seconds)
    omegaBField = magneticFieldRotationRPS * 2 * np.pi  # unit: rad/s

    # initialize key dataset
    raftLocations = np.zeros((numOfRafts, numOfTimeSteps, 2))  # in microns
    raftOrientations = np.zeros((numOfRafts, numOfTimeSteps))  # in deg
    raftRadii = np.ones(numOfRafts) * raftRadius  # in micron
    raftRotationSpeedsInRad = np.zeros((numOfRafts, numOfTimeSteps))  # in rad
    raftRelativeOrientationInDeg = np.zeros(
        (numOfRafts, numOfRafts, numOfTimeSteps))  # in deg, neighborID, raftID, frame#

    #    magDipoleForceOnAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    capillaryForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    hydrodynamicForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    magDipoleForceOffAxisTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    velocityTorqueCouplingTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    velocityMagFdTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    wallRepulsionTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    stochasticForceTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #    forceCurvatureTerm = np.zeros((numOfRafts, numOfTimeSteps, 2))
    #
    #    magneticFieldTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    magneticDipoleTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    capillaryTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))
    #    stochasticTorqueTerm = np.zeros((numOfRafts, numOfTimeSteps))

    currentStepNum = 0
    if initialPositionMethod == 1:
        # initialize the raft positions in the first frame, check pairwise ccdistance all above 2R
        paddingAroundArena = 20  # unit: radius
        ccDistanceMin = 2.5  # unit: radius
        raftLocations[:, currentStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                                arenaSize - raftRadius * paddingAroundArena,
                                                                (numOfRafts, 2))
        pairwiseDistances = scipy_distance.cdist(raftLocations[:, currentStepNum, :],
                                                 raftLocations[:, currentStepNum, :], 'euclidean')
        np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
        raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
        raftsToRelocate = np.unique(raftsToRelocate)

        while len(raftsToRelocate) > 0:
            raftLocations[raftsToRelocate, currentStepNum, :] = np.random.uniform(0 + raftRadius * paddingAroundArena,
                                                                                  arenaSize - raftRadius * paddingAroundArena,
                                                                                  (len(raftsToRelocate), 2))
            pairwiseDistances = scipy_distance.cdist(raftLocations[:, currentStepNum, :],
                                                     raftLocations[:, currentStepNum, :], 'euclidean')
            np.fill_diagonal(pairwiseDistances, raftRadius * ccDistanceMin + 1)
            raftsToRelocate, _ = np.nonzero(pairwiseDistances < raftRadius * ccDistanceMin)
            raftsToRelocate = np.unique(raftsToRelocate)

    elif initialPositionMethod == 2 or (initialPositionMethod == 3 and firstSpinSpeedFlag == 1):
        raftLocations[0, currentStepNum, :] = np.array([arenaSize / 2 + ccSeparationStarting / 2, arenaSize / 2])
        raftLocations[1, currentStepNum, :] = np.array([arenaSize / 2 - ccSeparationStarting / 2, arenaSize / 2])
        firstSpinSpeedFlag = 0
    elif initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
        raftLocations[0, currentStepNum, :] = lastPositionOfPreviousSpinSpeeds[0, :]
        raftLocations[1, currentStepNum, :] = lastPositionOfPreviousSpinSpeeds[1, :]

    raftOrientations[:, currentStepNum] = initialOrientation
    raftRotationSpeedsInRad[:, currentStepNum] = omegaBField

    outputFilename = 'Simulation_' + solverMethod + '_' + str(numOfRafts) + 'Rafts_' + str(
        magneticFieldRotationRPS).zfill(3) + 'rps_B' + str(magneticFieldStrength) + 'T_m' + str(
        magneticMomentOfOneRaft) + \
                     'Am2_capPeak' + str(capillaryPeakOffset) + '_edgeSmooth' + str(
        nearEdgeSmoothingThres) + '_torqueNoise' + str(stdOfTorqueNoise) + \
                     '_lubEqThres' + str(lubEqThreshold) + '_timeStep' + str(timeStepSize) + '_' + str(timeTotal) + 's'

    if outputVideo == 1:
        outputVideoName = outputFilename + '.mp4'
        fourcc = cv.VideoWriter_fourcc(*'DIVX')
        frameW, frameH, _ = blankFrameBGR.shape
        videoOut = cv.VideoWriter(outputVideoName, fourcc, outputFrameRate, (frameH, frameW), 1)

    for currentStepNum in progressbar.progressbar(np.arange(0, numOfTimeSteps - 1)):
        # currentStepNum = 0
        # looping over raft i,

        magneticFieldDirection = (
                                         magneticFieldRotationRPS * 360 * currentStepNum * timeStepSize) % 360  # neat trick to convert angles into [0, 360)

        raftLocationsOrientations = np.concatenate(
            (raftLocations[:, currentStepNum, :].flatten(), raftOrientations[:, currentStepNum]))

        sol = solve_ivp(Fun_drdt_dalphadt, (0, timeStepSize), raftLocationsOrientations, method=solverMethod)

        #        sol.y[np.logical_and((-sol.y < eps), (-sol.y > 0))] = 0

        raftLocations[:, currentStepNum + 1, :] = sol.y[0:numOfRafts * 2, -1].reshape(numOfRafts, 2)
        raftOrientations[:, currentStepNum + 1] = sol.y[numOfRafts * 2: numOfRafts * 3, -1]

        # draw for current frame
        if (outputImageSeq == 1 or outputVideo == 1) and (currentStepNum % intervalBetweenFrames == 0):
            currentFrameBGR = draw_rafts_rh_coord(blankFrameBGR.copy(),
                                                  np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                  np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_b_field_in_rh_coord(currentFrameBGR, magneticFieldDirection)
            currentFrameBGR = draw_cap_peaks_rh_coord(currentFrameBGR,
                                                      np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                      raftOrientations[:, currentStepNum], 6, capillaryPeakOffset,
                                                      np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_raft_orientations_rh_coord(currentFrameBGR,
                                                              np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                              raftOrientations[:, currentStepNum],
                                                              np.int64(raftRadii / scaleBar), numOfRafts)
            currentFrameBGR = draw_raft_num_rh_coord(currentFrameBGR,
                                                     np.int64(raftLocations[:, currentStepNum, :] / scaleBar),
                                                     numOfRafts)

            vector1To2SingleFrame = raftLocations[1, currentStepNum, :] - raftLocations[0, currentStepNum, :]
            distanceSingleFrame = np.sqrt(vector1To2SingleFrame[0] ** 2 + vector1To2SingleFrame[1] ** 2)
            phase1To2SingleFrame = np.arctan2(vector1To2SingleFrame[1], vector1To2SingleFrame[0]) * 180 / np.pi
            currentFrameBGR = draw_frame_info(currentFrameBGR, currentStepNum, distanceSingleFrame,
                                              raftOrientations[0, currentStepNum], magneticFieldDirection,
                                              raftRelativeOrientationInDeg[0, 1, currentStepNum])

            if outputImageSeq == 1:
                outputImageName = outputFilename + '_' + str(currentStepNum + 1).zfill(7) + '.jpg'
                cv.imwrite(outputImageName, currentFrameBGR)
            if outputVideo == 1:
                videoOut.write(np.uint8(currentFrameBGR))

    #        if distanceSingleFrame > 950:
    #            break

    if outputVideo == 1:
        videoOut.release()

    tempShelf = shelve.open(outputFilename)
    for key in listOfVariablesToSave:
        try:
            tempShelf[key] = globals()[key]
        except TypeError:
            #
            # __builtins__, tempShelf, and imported modules can not be shelved.
            #
            # print('ERROR shelving: {0}'.format(key))
            pass
    tempShelf.close()

#    lastPositionOfPreviousSpinSpeeds[:,:] = raftLocations[:,currentStepNum,:]
#    lastOmegaOfPreviousSpinSpeeds[:] = raftRotationSpeedsInRad[:, currentStepNum]


# %% load simulated data in one main folder

rootFolderNameforSimulation = r'D:\SimulationFolder'
os.chdir(rootFolderNameforSimulation)
rootFolderTreeGen = os.walk(rootFolderNameforSimulation)
_, mainFolders, _ = next(rootFolderTreeGen)

mainFolderID = 38
os.chdir(mainFolders[mainFolderID])

dataFileList = glob.glob('*.dat')
dataFileList.sort()

mainDataList = []
variableListsForAllMainData = []

for dataID in range(10, 11):  # len(dataFileList)):
    dataFileToLoad = dataFileList[dataID].partition('.dat')[0]

    tempShelf = shelve.open(dataFileToLoad)
    variableListOfOneMainDataFile = list(tempShelf.keys())

    expDict = {}
    for key in tempShelf:
        try:
            expDict[key] = tempShelf[key]
        except TypeError:
            pass

    tempShelf.close()
    mainDataList.append(expDict)
    variableListsForAllMainData.append(variableListOfOneMainDataFile)

# %% pairwise data treatment and output to csv

samplingRate = 1 / timeStepSize  # unit fps
diameterOfRaftInMicron = 300  # micron
startOfSamplingStep = 49000
# diameterOfRaftInPixel = 146 # pixel 124 for 2x mag, 146 for 2.5x object,
# scaleBar = diameterOfRaftInMicron/diameterOfRaftInPixel # micron per pixel. 300 micron = 124 pixel -> 2x objective, 300 micron = 146 pixel -> 2.5x objective

# initialize data frames
varsForMainData = ['mainFolderName', 'experimentName', 'batchNum', 'magneticFieldRotationRPS',
                   'distancesMean', 'distancesSTD', 'orbitingSpeedsMean', 'orbitingSpeedsSTD',
                   'raft1SpinSpeedsMean', 'raft1SpinSpeedsSTD', 'raft2SpinSpeedsMean', 'raft2SpinSpeedsSTD']
dfMainData = pd.DataFrame(columns=varsForMainData, index=range(len(mainDataList)))

dfFFTDist = pd.DataFrame(columns=['fDistances'])

dfFFTOrbitingSpeeds = pd.DataFrame(columns=['fOrbitingSpeeds'])

dfFFTRaft1Spin = pd.DataFrame(columns=['fRaft1SpinSpeeds'])

dfFFTRaft2Spin = pd.DataFrame(columns=['fRaft2SpinSpeeds'])

for dataID in range(len(mainDataList)):
    raft1Locations = mainDataList[dataID]['raftLocations'][0, startOfSamplingStep:, :]
    raft2Locations = mainDataList[dataID]['raftLocations'][1, startOfSamplingStep:, :]

    vector1To2 = raft2Locations - raft1Locations

    distances = np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)
    distancesMean = distances.mean()  #
    distancesSTD = np.std(distances)

    fDistances, pDistances = fft_distances(samplingRate, distances)

    phase1To2 = np.arctan2(vector1To2[:, 1], vector1To2[:,
                                             0]) * 180 / np.pi  # note that the sign of y is flipped, so as to keep the coordination in the Rhino convention
    phasesAjusted = adjust_phases(phase1To2)
    orbitingSpeeds = np.gradient(phasesAjusted) * samplingRate / 180 * np.pi
    orbitingSpeedsMean = orbitingSpeeds.mean()
    orbitingSpeedsSTD = orbitingSpeeds.std()

    fOrbitingSpeeds, pOrbitingSpeeds = fft_distances(samplingRate, orbitingSpeeds)

    raft1Orientations = mainDataList[dataID]['raftOrientations'][0, startOfSamplingStep:]
    raft2Orientations = mainDataList[dataID]['raftOrientations'][1, startOfSamplingStep:]
    raft1OrientationsAdjusted = adjust_phases(raft1Orientations)
    raft2OrientationsAdjusted = adjust_phases(raft2Orientations)
    raft1SpinSpeeds = np.gradient(raft1OrientationsAdjusted) * samplingRate / 360
    raft2SpinSpeeds = np.gradient(raft2OrientationsAdjusted) * samplingRate / 360
    raft1SpinSpeedsMean = raft1SpinSpeeds.mean()
    raft2SpinSpeedsMean = raft2SpinSpeeds.mean()
    raft1SpinSpeedsSTD = raft1SpinSpeeds.std()
    raft2SpinSpeedsSTD = raft2SpinSpeeds.std()

    fRaft1SpinSpeeds, pRaft1SpinSpeeds = fft_distances(samplingRate, raft1SpinSpeeds)
    fRaft2SpinSpeeds, pRaft2SpinSpeeds = fft_distances(samplingRate, raft2SpinSpeeds)

    # store in dataframes
    dfMainData.loc[dataID, 'mainFolderName'] = mainFolders[mainFolderID]
    #    if mainDataList[dataID]['isVideo'] == 0:
    #        dfMainData.loc[dataID,'experimentName'] = mainDataList[dataID]['subfolders'][mainDataList[dataID]['expID']]
    #    elif mainDataList[dataID]['isVideo'] == 1:
    #        dfMainData.loc[dataID,'experimentName'] = mainDataList[dataID]['videoFileList'][mainDataList[dataID]['expID']]
    #    dfMainData.loc[dataID,'batchNum'] = mainDataList[dataID]['batchNum']
    dfMainData.loc[dataID, 'magneticFieldRotationRPS'] = - mainDataList[dataID]['magneticFieldRotationRPS']
    dfMainData.loc[dataID, 'distancesMean'] = distancesMean - diameterOfRaftInMicron
    dfMainData.loc[dataID, 'distancesSTD'] = distancesSTD
    dfMainData.loc[dataID, 'orbitingSpeedsMean'] = -orbitingSpeedsMean
    dfMainData.loc[dataID, 'orbitingSpeedsSTD'] = orbitingSpeedsSTD
    dfMainData.loc[dataID, 'raft1SpinSpeedsMean'] = -raft1SpinSpeedsMean
    dfMainData.loc[dataID, 'raft1SpinSpeedsSTD'] = raft1SpinSpeedsSTD
    dfMainData.loc[dataID, 'raft2SpinSpeedsMean'] = -raft2SpinSpeedsMean
    dfMainData.loc[dataID, 'raft2SpinSpeedsSTD'] = raft2SpinSpeedsSTD

    if len(dfFFTDist) == 0:
        dfFFTDist['fDistances'] = fDistances
    #    colName = str(mainDataList[dataID]['batchNum']) + '_' + str(mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)
    colName = str(-mainDataList[dataID]['magneticFieldRotationRPS']).zfill(4)
    dfFFTDist[colName] = pDistances

    if len(dfFFTOrbitingSpeeds) == 0:
        dfFFTOrbitingSpeeds['fOrbitingSpeeds'] = fOrbitingSpeeds
    dfFFTOrbitingSpeeds[colName] = pOrbitingSpeeds

    if len(dfFFTRaft1Spin) == 0:
        dfFFTRaft1Spin['fRaft1SpinSpeeds'] = fRaft1SpinSpeeds
    dfFFTRaft1Spin[colName] = pRaft1SpinSpeeds

    if len(dfFFTRaft2Spin) == 0:
        dfFFTRaft2Spin['fRaft2SpinSpeeds'] = fRaft2SpinSpeeds
    dfFFTRaft2Spin[colName] = pRaft2SpinSpeeds

dfMainData = dfMainData.infer_objects()
# dfMainData.sort_values(by = ['batchNum','magneticFieldRotationRPS'], ascending = [True, False], inplace = True)
dfMainData.sort_values(by=['magneticFieldRotationRPS'], ascending=[False], inplace=True)

dfFFTDist = dfFFTDist.infer_objects()
dfFFTOrbitingSpeeds = dfFFTOrbitingSpeeds.infer_objects()
dfFFTRaft1Spin = dfFFTRaft1Spin.infer_objects()
dfFFTRaft2Spin = dfFFTRaft2Spin.infer_objects()

dfFFTDist = dfFFTDist.reindex(sorted(dfFFTDist.columns, reverse=True), axis='columns')
dfFFTOrbitingSpeeds = dfFFTOrbitingSpeeds.reindex(sorted(dfFFTOrbitingSpeeds.columns, reverse=True), axis='columns')
dfFFTRaft1Spin = dfFFTRaft1Spin.reindex(sorted(dfFFTRaft1Spin.columns, reverse=True), axis='columns')
dfFFTRaft2Spin = dfFFTRaft2Spin.reindex(sorted(dfFFTRaft2Spin.columns, reverse=True), axis='columns')

dfMainData.plot.scatter(x='magneticFieldRotationRPS', y='distancesMean')

# output to csv files
mainDataFileName = mainFolders[mainFolderID]
colNames = ['batchNum', 'magneticFieldRotationRPS',
            'distancesMean', 'distancesSTD', 'orbitingSpeedsMean', 'orbitingSpeedsSTD',
            'raft1SpinSpeedsMean', 'raft1SpinSpeedsSTD', 'raft2SpinSpeedsMean', 'raft2SpinSpeedsSTD']
dfMainData.to_csv(mainDataFileName + '.csv', index=False, columns=colNames)

# BFieldStrength = '10mT'
BFieldStrength = str(magneticFieldStrength * 1000).zfill(4) + 'mT'
dfFFTDist.to_csv('fft_' + BFieldStrength + '_distance.csv', index=False)
dfFFTOrbitingSpeeds.to_csv('fft_' + BFieldStrength + '_orbitingSpeeds.csv', index=False)
dfFFTRaft1Spin.to_csv('fft_' + BFieldStrength + '_raft1SpinSpeeds.csv', index=False)
dfFFTRaft2Spin.to_csv('fft_' + BFieldStrength + '_raft2SpinSpeeds.csv', index=False)

# testing the random distribution sampling:
# mu, sigma = 0, 0.01 # mean and standard deviation
# s = np.random.normal(mu, sigma, 10000)
# count, bins, ignored = plt.hist(s, 30, density=True)
# plt.plot(bins, 1/(sigma * np.sqrt(2 * np.pi)) * np.exp( - (bins - mu)**2 / (2 * sigma**2) ), linewidth=2, color='r')
# plt.show()

# %% load one specific simulated data and look at the results

dataID = 0

variableListFromSimulatedFile = list(mainDataList[dataID].keys())

for key, value in mainDataList[dataID].items():  # loop through key-value pairs of python dictionary
    globals()[key] = value

# data treatment
startOfSamplingStep = 0  # 0, 10000
samplingRate = 1 / timeStepSize  #
raft1Locations = raftLocations[0, startOfSamplingStep:, :]  # unit: micron
raft2Locations = raftLocations[1, startOfSamplingStep:, :]  # unit: micron

vector1To2 = raft2Locations - raft1Locations  # unit: micron
distances = np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)  # unit micron, pairwise ccDistances
distancesMean = distances.mean()
distancesSTD = distances.std()

distancesDownSampled = distances[::100]

fDistances, pDistances = fft_distances(samplingRate, distances)

phase1To2 = np.arctan2(vector1To2[:, 1], vector1To2[:,
                                         0]) * 180 / np.pi  # note that the sign of y is flipped, so as to keep the coordination in the Rhino convention
phasesAjusted = adjust_phases(phase1To2)
orbitingSpeeds = np.gradient(phasesAjusted) * samplingRate / 180 * np.pi
orbitingSpeedsMean = orbitingSpeeds.mean()
orbitingSpeedsSTD = orbitingSpeeds.std()

fOrbitingSpeeds, pOrbitingSpeeds = fft_distances(samplingRate, orbitingSpeeds)

raft1Orientations = raftOrientations[0, startOfSamplingStep:]
raft2Orientations = raftOrientations[1, startOfSamplingStep:]
raft1OrientationsAdjusted = adjust_phases(raft1Orientations)
raft2OrientationsAdjusted = adjust_phases(raft2Orientations)
raft1SpinSpeeds = np.gradient(raft1OrientationsAdjusted) * samplingRate / 360  # unit: rps
raft2SpinSpeeds = np.gradient(raft2OrientationsAdjusted) * samplingRate / 360  # unit: rps
raft1SpinSpeedsMean = raft1SpinSpeeds.mean()
raft2SpinSpeedsMean = raft2SpinSpeeds.mean()
raft1SpinSpeedsSTD = raft1SpinSpeeds.std()
raft2SpinSpeedsSTD = raft2SpinSpeeds.std()

fRaft1SpinSpeeds, pRaft1SpinSpeeds = fft_distances(samplingRate, raft1SpinSpeeds)
fRaft2SpinSpeeds, pRaft2SpinSpeeds = fft_distances(samplingRate, raft2SpinSpeeds)

# plotting analyzed results
# comparison of force terms
fig, ax = plt.subplots(ncols=1, nrows=1)
vector1To2X_Unitized = vector1To2[:, 0] / np.sqrt(vector1To2[:, 0] ** 2 + vector1To2[:, 1] ** 2)
ax.plot(magDipoleForceOnAxisTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
        label='magnetic-dipole-force velocity term on raft2 / vector1To2')
ax.plot(capillaryForceTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
        label='capillary-force velocity term on raft2 / vector1To2')
ax.plot(hydrodynamicForceTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
        label='hydrodynamic-force velocity term on raft2 / vector1To2')
ax.plot(wallRepulsionTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
        label='wall-repulsion velocity term on raft2 / vector1To2')
ax.plot(forceCurvatureTerm[1, startOfSamplingStep:, 0] / vector1To2X_Unitized * timeStepSize, '-',
        label='force-curvature velocity term on raft2 / vector1To2')

# ax.plot(magDipoleForceOnAxisTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-', label = 'magnetic-dipole-force velocity term on raft1 / vector1To2')
# ax.plot(capillaryForceTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-', label = 'capillary-force velocity term on raft1 / vector1To2')
# ax.plot(hydrodynamicForceTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-', label = 'hydrodynamic-force velocity term on raft1 / vector1To2')
# ax.plot(wallRepulsionTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-', label = 'wall-repulsion velocity term on raft1 / vector1To2')
# ax.plot(forceCurvatureTerm[0,startOfSamplingStep:,0]/vector1To2[:,0], '-', label = 'force-curvature velocity term on raft1 / vector1To2')
ax.set_xlabel('time step number', size=20)
ax.set_ylabel('displacement along vector1To2 (um)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting distances between rafts vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arange(numOfTimeSteps) * timeStepSize, distances, '-o', label='c')
ax.set_xlabel('Time (s)', size=20)
ax.set_ylabel('ccdistances between rafts (micron)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of distances
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fDistances[1:], pDistances[1:], '-o', label='c')
ax.set_xlabel('fDistances (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting orbiting speeds vs time
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(orbitingSpeeds, '-o', label='orbiting speeds calculated from orientation')
ax.set_xlabel('Frames(Time)', size=20)
ax.set_ylabel('orbiting speeds in rad/s', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of orbiting speed
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fOrbitingSpeeds[1:], pOrbitingSpeeds[1:], '-o', label='c')
ax.set_xlabel('fOrbitingSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# comparison of torque terms
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(magneticFieldTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
        label='magnetic field torque term of raft 2')
ax.plot(magneticDipoleTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
        label='magnetic dipole torque term of raft 2')
ax.plot(capillaryTorqueTerm[1, startOfSamplingStep:] / np.pi * 180 * timeStepSize, '-',
        label='capillary torque term of raft 2')
# ax.plot(magneticFieldTorqueTerm[0,startOfSamplingStep:], '-', label = 'magnetic field torque term of raft 1')
# ax.plot(magneticDipoleTorqueTerm[0,startOfSamplingStep:], '-', label = 'magnetic dipole torque term of raft 1')
# ax.plot(capillaryTorqueTerm[0,startOfSamplingStep:], '-', label = 'capillary torque term of raft 1')
ax.set_xlabel('time step number', size=20)
ax.set_ylabel('rotation d_alpha (deg)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft relative orientation phi_ji
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raftRelativeOrientationInDeg[0, 1, :], '-o', label='relative orientation of raft 2 and neighbor 1')
# ax.plot(raftRelativeOrientationInDeg[1, 0, :],'-o', label = 'relative orientation of raft 1 and neighbor 2')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('orientation angles', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# the angles between the magnetic dipole moment and the magnetic field
# ref: magneticFieldTorqueTerm[raftID, currentStepNum] = tb * magneticFieldStrength * magneticMomentOfOneRaft * np.sin(np.deg2rad(magneticFieldDirection - raftOrientations[raftID,currentStepNum])) * 1e6 /(8*np.pi*miu*R**3)

miu = 1e-15  # dynamic viscosity of water; unit conversion: 1e-3 Pa.s = 1e-3 N.s/m^2 = 1e-15 N.s/um^2
R = 1.5e2  # unit: micron

fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(np.arcsin((magneticFieldTorqueTerm[0, startOfSamplingStep:] * (8 * np.pi * miu * R ** 3)) / (
        magneticFieldStrength * magneticMomentOfOneRaft * 1e6)) / np.pi * 180, '-',
        label='the angle between B and m')
ax.set_xlabel('time step number', size=20)
ax.set_ylabel('angle (deg)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft orientations vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1Orientations, '-o', label='raft 1 orientation before adjustment')
ax.plot(raft2Orientations, '-o', label='raft 2 orientation before adjustment')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('orientation angles', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft orientations adjusted vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1OrientationsAdjusted, '-o', label='raft 1 orientation adjusted')
ax.plot(raft2OrientationsAdjusted, '-o', label='raft 2 orientation adjusted')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('orientation angles adjusted', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting raft spin speeds vs frame# (time)
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(raft1SpinSpeeds, '-', label='raft 1 spin speeds')
ax.plot(raft2SpinSpeeds, '-', label='raft 2 spin speeds')
# ax.plot(np.deg2rad(adjust_phases(raftOrientations[0,startOfSamplingStep+1:]) - adjust_phases(raftOrientations[0,startOfSamplingStep:-1]))/timeStepSize/(2*np.pi), '-')
ax.set_xlabel('Steps(Time)', size=20)
ax.set_ylabel('spin speeds (rps)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of spin speeds of raft 1
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fRaft1SpinSpeeds[1:], pRaft1SpinSpeeds[1:], '-o', label='c')
ax.set_xlabel('fRaft1SpinSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# plotting the fft of spin speeds of raft 2
fig, ax = plt.subplots(ncols=1, nrows=1)
ax.plot(fRaft2SpinSpeeds[1:], pRaft2SpinSpeeds[1:], '-o', label='c')
ax.set_xlabel('fRaft2SpinSpeeds (Hz)', size=20)
ax.set_ylabel('Power P1 (a.u.)', size=20)
ax.set_title('Simulation at {}rps'.format(magneticFieldRotationRPS))
ax.legend()
plt.show()

# %% code to be deleted:

for raftID in np.arange(numOfRafts):
    # raftID = 0
    R = raftRadii[raftID]  # unit: micron
    ri = raftLocations[raftID, currentStepNum, :]  # unit: micron

    d_ri = np.zeros(2)
    d_alpha = np.zeros(1)

    # force cuvature term
    if forceDueToCurvature != 0:
        ricenter = centerOfArena - ri
        ricenterNorm = np.sqrt(ricenter[0] ** 2 + ricenter[1] ** 2)
        ricenterUnitized = ricenter / ricenterNorm
        forceCurvatureTerm[raftID, currentStepNum, :] = forceDueToCurvature / (6 * piMiuR) * ricenterUnitized

    # magnetic field torque:
    magneticFieldTorque = magneticFieldStrength * magneticMomentOfOneRaft * np.sin(
        np.deg2rad(magneticFieldDirection - raftOrientations[raftID, currentStepNum]))  # unit: N.um
    magneticFieldTorqueTerm[raftID, currentStepNum] = tb * magneticFieldTorque * 1e6 / (
            8 * piMiuR * R ** 2)  # unit: 1/s

    # looping over all neighbors of raft i; for pairwise or small number of rafts, just loop over all the rest of rafts

    rji_eeDist_smallest = R;  # initialize
    for neighborID in np.arange(numOfRafts):
        if neighborID == raftID:
            continue
        rj = raftLocations[neighborID, currentStepNum, :]  # unit: micron
        rji = ri - rj  # unit: micron
        rjiNorm = np.sqrt(rji[0] ** 2 + rji[1] ** 2)  # unit: micron
        rji_eeDist = rjiNorm - 2 * R  # unit: micron
        rjiUnitized = rji / rjiNorm  # unit: micron
        rjiUnitizedCrossZ = np.asarray((rjiUnitized[1], -rjiUnitized[0]))
        phi_ji = map_angles_from0to360(np.arctan2(rji[1], rji[0]) * 180 / np.pi - raftOrientations[
            raftID, currentStepNum])  # unit: deg; assuming both rafts's orientations are the same
        raftRelativeOrientationInDeg[neighborID, raftID, currentStepNum] = phi_ji

        # force terms:
        if currentStepNum > 1:
            omega = raftRotationSpeedsInRad[neighborID, currentStepNum - 1]
        elif currentStepNum == 0:
            if initialPositionMethod == 3 and firstSpinSpeedFlag == 0:
                omega = lastOmegaOfPreviousSpinSpeeds[neighborID]
            else:
                omega = omegaBField

        if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold:
            magDipoleForceOnAxisTerm[raftID, currentStepNum, :] = magDipoleForceOnAxisTerm[raftID, currentStepNum,
                                                                  :] + cm * magDpForceOnAxis[
                                                                      int(rji_eeDist), int(phi_ji)] * rjiUnitized / (
                                                                          6 * piMiuR)  # unit: um/s
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            magDipoleForceOnAxisTerm[raftID, currentStepNum, :] = magDipoleForceOnAxisTerm[raftID, currentStepNum,
                                                                  :] + cm * lubA[
                                                                      int(rji_eeDist * lubCoeffScaleFactor)] * \
                                                                  magDpForceOnAxis[int(rji_eeDist), int(
                                                                      phi_ji)] * rjiUnitized / miu  # unit: um/s
        elif rji_eeDist < 0:
            magDipoleForceOnAxisTerm[raftID, currentStepNum, :] = magDipoleForceOnAxisTerm[raftID, currentStepNum,
                                                                  :] + cm * lubA[0] * magDpForceOnAxis[
                                                                      0, int(phi_ji)] * rjiUnitized / miu  # unit: um/s

        if rji_eeDist < 1000 and rji_eeDist >= lubEqThreshold:
            capillaryForceTerm[raftID, currentStepNum, :] = capillaryForceTerm[raftID, currentStepNum, :] + cc * \
                                                            capillaryForcesDistancesAsRows[
                                                                int(rji_eeDist), int(phi_ji)] * rjiUnitized / (
                                                                    6 * piMiuR)  # unit: um/s
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            capillaryForceTerm[raftID, currentStepNum, :] = capillaryForceTerm[raftID, currentStepNum, :] + cc * lubA[
                int(rji_eeDist * lubCoeffScaleFactor)] * capillaryForcesDistancesAsRows[int(rji_eeDist), int(
                phi_ji)] * rjiUnitized / miu  # unit: um/s
        elif rji_eeDist < 0:
            capillaryForceTerm[raftID, currentStepNum, :] = capillaryForceTerm[raftID, currentStepNum, :] + cc * lubA[
                0] * capillaryForcesDistancesAsRows[0, int(phi_ji)] * rjiUnitized / miu  # unit: um/s

        if rji_eeDist >= lubEqThreshold:
            hydrodynamicForceTerm[raftID, currentStepNum, :] = hydrodynamicForceTerm[raftID, currentStepNum,
                                                               :] + ch * 1e-6 * densityOfWater * omega ** 2 * R ** 7 * rji / rjiNorm ** 4 / (
                                                                       6 * piMiuR)  # unit: um/s; 1e-6 is used to convert the implicit m to um in Newton in miu
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            hydrodynamicForceTerm[raftID, currentStepNum, :] = hydrodynamicForceTerm[raftID, currentStepNum, :] + ch * \
                                                               lubA[int(rji_eeDist * lubCoeffScaleFactor)] * (
                                                                       1e-6 * densityOfWater * omega ** 2 * R ** 7 / rjiNorm ** 3) * rjiUnitized / miu  # unit: um/s

        if rji_eeDist < 10000 and rji_eeDist >= lubEqThreshold:
            magDipoleForceOffAxisTerm[raftID, currentStepNum, :] = magDipoleForceOffAxisTerm[raftID, currentStepNum,
                                                                   :] + magDpForceOffAxis[int(rji_eeDist), int(
                phi_ji)] * rjiUnitizedCrossZ / (6 * piMiuR)
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            magDipoleForceOffAxisTerm[raftID, currentStepNum, :] = magDipoleForceOffAxisTerm[raftID, currentStepNum,
                                                                   :] + lubB[int(rji_eeDist * lubCoeffScaleFactor)] * \
                                                                   magDpForceOffAxis[int(rji_eeDist), int(
                                                                       phi_ji)] * rjiUnitizedCrossZ / miu  # unit: um/s
        elif rji_eeDist < 0:
            magDipoleForceOffAxisTerm[raftID, currentStepNum, :] = magDipoleForceOffAxisTerm[raftID, currentStepNum,
                                                                   :] + lubB[0] * magDpForceOffAxis[0, int(
                phi_ji)] * rjiUnitizedCrossZ / miu  # unit: um/s

        if rji_eeDist >= lubEqThreshold:
            velocityTorqueCouplingTerm[raftID, currentStepNum, :] = velocityTorqueCouplingTerm[raftID, currentStepNum,
                                                                    :] - R ** 3 * omega * rjiUnitizedCrossZ / (
                                                                            rjiNorm ** 2)  # unit: um/s
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            velocityMagFdTorqueTerm[raftID, currentStepNum, :] = velocityMagFdTorqueTerm[raftID, currentStepNum, :] + \
                                                                 lubC[int(
                                                                     rji_eeDist * lubCoeffScaleFactor)] * magneticFieldTorque * 1e6 * rjiUnitizedCrossZ / miu  # unit: um/s
        elif rji_eeDist < 0:
            velocityMagFdTorqueTerm[raftID, currentStepNum, :] = velocityMagFdTorqueTerm[raftID, currentStepNum, :] + \
                                                                 lubC[
                                                                     0] * magneticFieldTorque * 1e6 * rjiUnitizedCrossZ / miu  # unit: um/s

        #                if rji_eeDist >= lubEqThreshold and currentStepNum > 1:
        #                    prev_drdt = (raftLocations[raftID,currentStepNum,:] -  raftLocations[raftID,currentStepNum-1,:]) / timeStepSize
        #                    stochasticTerm[raftID, currentStepNum,:] =  stochasticTerm[raftID, currentStepNum,:] + np.sqrt(prev_drdt[0]**2 + prev_drdt[1]**2) * np.random.normal(0, stdOfFluctuationTerm, 1) * rjiUnitized

        if rji_eeDist < 0:
            wallRepulsionTerm[raftID, currentStepNum, :] = wallRepulsionTerm[raftID, currentStepNum, :] + (
                -rji_eeDist) * rjiUnitized / timeStepSize

        # torque terms:
        if rji_eeDist < lubEqThreshold and rji_eeDist < rji_eeDist_smallest:
            rji_eeDist_smallest = rji_eeDist
            if rji_eeDist_smallest >= 0:
                magneticFieldTorqueTerm[raftID, currentStepNum] = lubG[int(
                    rji_eeDist_smallest)] * magneticFieldTorque * 1e6 / miu  # unit: 1/s
            elif rji_eeDist_smallest < 0:
                magneticFieldTorqueTerm[raftID, currentStepNum] = lubG[0] * magneticFieldTorque * 1e6 / miu  # unit: 1/s

        if rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            magneticFieldTorqueTerm[raftID, currentStepNum] = lubG[int(
                rji_eeDist)] * magneticFieldTorque * 1e6 / miu  # unit: 1/s
        elif rji_eeDist < 0:
            magneticFieldTorqueTerm[raftID, currentStepNum] = lubG[0] * magneticFieldTorque * 1e6 / miu  # unit: 1/s

        if rji_eeDist < 10000 and rji_eeDist >= 0:
            magneticDipoleTorqueTerm[raftID, currentStepNum] = magneticDipoleTorqueTerm[raftID, currentStepNum] + tm * \
                                                               magDpTorque[int(rji_eeDist), int(phi_ji)] * 1e6 / (
                                                                       8 * piMiuR * R ** 2)
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            magneticDipoleTorqueTerm[raftID, currentStepNum] = magneticDipoleTorqueTerm[raftID, currentStepNum] + tm * \
                                                               lubH[int(rji_eeDist * lubCoeffScaleFactor)] * \
                                                               magDpTorque[int(rji_eeDist), int(
                                                                   phi_ji)] * 1e6 / miu  # unit: 1/s
        elif rji_eeDist < 0:
            magneticDipoleTorqueTerm[raftID, currentStepNum] = magneticDipoleTorqueTerm[raftID, currentStepNum] + tm * \
                                                               lubH[0] * magDpTorque[
                                                                   0, int(phi_ji)] * 1e6 / miu  # unit: 1/s

        if rji_eeDist < 1000 and rji_eeDist >= 0:
            capillaryTorqueTerm[raftID, currentStepNum] = capillaryTorqueTerm[raftID, currentStepNum] + tc * \
                                                          capillaryTorquesDistancesAsRows[
                                                              int(rji_eeDist), int(phi_ji)] * 1e6 / (
                                                                  8 * piMiuR * R ** 2)  # unit: 1/s
        elif rji_eeDist < lubEqThreshold and rji_eeDist >= 0:
            capillaryTorqueTerm[raftID, currentStepNum] = capillaryTorqueTerm[raftID, currentStepNum] + tc * lubH[
                int(rji_eeDist * lubCoeffScaleFactor)] * capillaryTorquesDistancesAsRows[
                                                              int(rji_eeDist), int(phi_ji)] * 1e6 / miu  # unit: 1/s
        elif rji_eeDist < 0:
            capillaryTorqueTerm[raftID, currentStepNum] = capillaryTorqueTerm[raftID, currentStepNum] + tc * lubH[0] * \
                                                          capillaryTorquesDistancesAsRows[
                                                              0, int(phi_ji)] * 1e6 / miu  # unit: 1/s

    # update position
    d_ri = (magDipoleForceOnAxisTerm[raftID, currentStepNum, :] + capillaryForceTerm[raftID, currentStepNum,
                                                                  :] + hydrodynamicForceTerm[raftID, currentStepNum,
                                                                       :] + magDipoleForceOffAxisTerm[raftID,
                                                                            currentStepNum,
                                                                            :] + velocityTorqueCouplingTerm[raftID,
                                                                                 currentStepNum,
                                                                                 :] + velocityMagFdTorqueTerm[raftID,
                                                                                      currentStepNum,
                                                                                      :] + stochasticTerm[raftID,
                                                                                           currentStepNum,
                                                                                           :] + wallRepulsionTerm[
                                                                                                raftID, currentStepNum,
                                                                                                :] + forceCurvatureTerm[
                                                                                                     raftID,
                                                                                                     currentStepNum,
                                                                                                     :]) * timeStepSize

    # update angles and raftRotationSpeedsInRad
    raftRotationSpeedsInRad[raftID, currentStepNum] = (
            magneticFieldTorqueTerm[raftID, currentStepNum] + magneticDipoleTorqueTerm[raftID, currentStepNum] +
            capillaryTorqueTerm[raftID, currentStepNum])
    d_alpha = raftRotationSpeedsInRad[raftID, currentStepNum] / np.pi * 180 * timeStepSize  # in deg before assembly

    # store the location and orientation for the next time step
    raftLocations[raftID, currentStepNum + 1, :] = raftLocations[raftID, currentStepNum, :] + d_ri
    raftOrientations[raftID, currentStepNum + 1] = map_angles_from0to360(
        raftOrientations[raftID, currentStepNum] + d_alpha)
