#!/usr/bin/python3
import numpy as np
import os, sys
import docker
import pandas as pd

import shutil
import subprocess
import shlex
import argparse
import signal
import json

from plot_coefficients import (plot_coefficients, get_coefficients_df, get_aerodynamic_coefficients)

#from PyFoam.Applications.PlotWatcher import PlotWatcher

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)


def input_with_timeout(msg):
    try:
        print(msg)
        ans = input()
        return ans
    except RuntimeError as e:
        print(e)
        return "n"


def interrupted(signum, frame):
    "called when read times out"
    print("timeout")
    raise RuntimeError("timeout!")
signal.signal(signal.SIGALRM, interrupted)


def write_bashrc_file(filename, bashrc):
    f = open(filename, "r")
    contents = f.readlines()
    f.close()
    contents.insert(1, "source {}\n".format(bashrc))
    f = open(filename, "w")
    contents = "".join(contents)
    f.write(contents)
    f.close()


def read_aorfoil(filename):
    ignore_head = 1
    coords = []
    with open(filename, "r") as fid:
        for line in fid.readlines()[ignore_head:]:
            data = line.split()
            coords.append((float(data[0]), float(data[1])))

    # Se asegura de que el punto inicial este repetido al final de la secuencia
    if coords[0] != coords[-1]:
        coords.append(coords[0])
    return coords


def apply_rotation(coords, angle, center=(0.25,0)):
    center = np.array(center)
    theta = np.radians(angle)
    c, s = np.cos(theta), np.sin(theta)
    R = np.array(((c,-s), (s, c))).T
    coords_rotated = []
    for point in coords:
        vector = point - center
        vector_rotated = R.dot(vector)
        coords_rotated.append(tuple(vector_rotated + center))
    return coords_rotated


def find_normal(p1, p2):
    dx = p2[0] - p1[0]
    dy = p2[1] - p1[1]
    norma = np.sqrt(dx**2+dy**2)
    return np.array([dy/norma, -dx/norma])


def offset_point(p1,p2,p3, offset):
    normal = np.array([0,0])
    if p1==p2:
        normal = find_normal(p2,p3)
    elif p2==p3:
        normal = find_normal(p1,p2)
    else:
        normal = 0.5*(find_normal(p1,p2)+find_normal(p2,p3))
    return tuple(np.array(p2)+offset*normal)


def offset_coords(coords):
    coords_offset = []
    offset_min = 0.01
    offset_slope = 0.03
    offset_boundary = 0.3
    for i in range(len(coords)):
        x = coords_rotated[i][0]
        if x < offset_boundary:
            offset = offset_min
        else:
            offset = offset_min + (x-offset_boundary)*offset_slope
        if i == 0:
            coords_offset.append(offset_point(coords[i], coords[i], coords[i+1], offset))
        elif i==len(coords_offset):
            coords_offset.append(offset_point(coords[i-1], coords[i], coords[i], offset))
        else:
            coords_offset.append(offset_point(coords[i-1], coords[i], coords[i+1], offset))
    return coords_offset


def derrotate_and_normalize_coordinates(coords):
    x, y = zip(*coords)
    idx_min = np.argmin(x)
    idx_max = np.argmax(x)
    dy = y[idx_max] - y[idx_min]
    dx = x[idx_max] - x[idx_min]
    m = dy / dx
    chord = np.sqrt(dy ** 2 + dx ** 2)
    angle = np.arctan(-m)
    offset = (x[idx_min] + 0.25 * dx, y[idx_min] + m * 0.25 * dx)
    coods_derotated = apply_rotation(coords, -np.degrees(angle), center=offset)
    coords_offset_normalized = []
    idx_LE = np.argmin([point[0] for point in coods_derotated])
    LE_point = coods_derotated[idx_LE]
    for point in coods_derotated:
        coords_offset_normalized.append((1 / chord * (point[0] - LE_point[0]), 1 / chord * (point[1] - LE_point[1])))
    return coords_offset_normalized, angle, chord


def write_geo(output_filename, coords_rotated, lc=0.005, largo=10, alto=8):
    lc_name = "%s_lc" % output_filename[0:3]
    ancho = 1
    startpoint = 1000
    with open(output_filename, "w") as fid:
        fid.write("%s = %f;\n" % (lc_name, lc))
        j = startpoint
        for x, y in coords_rotated:
            outputline = "Point(%i) = { %8.8f, %8.8f, 0.0, %s};\n" % (j, x, y, lc_name)
            j = j + 1
            fid.write(outputline)
        fid.write("Spline(%i) = {%i:%i,%i};\n" % (startpoint, startpoint, j, startpoint))
        k = j + 1
        j = j + 1
        fid.write("//+\n")
        fid.write("Point(%i) = {-%1.4f, -%1.4f, 0, %1.4f};\n" % (j, 0.4 * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {%1.4f, -%1.4f, 0, %1.4f};\n" % (j, (1 - 0.4) * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {%1.4f, %1.4f, 0, %1.4f};\n" % (j, (1 - 0.4) * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {-%1.4f, %1.4f, 0, %1.4f};\n" % (j, 0.4 * largo, 0.5 * alto, ancho))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 1, k, k + 1))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 2, k + 1, k + 2))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 3, k + 2, k + 3))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 4, k + 3, k))
        fid.write("Line Loop(1) = {%i, %i, %i, %i};\n" % tuple(range(startpoint + 1, startpoint + 5)))
        fid.write("Line Loop(2) = {%i};\n" % startpoint)
        fid.write("""Surface(10) = {1, 2};
    TwoDimSurf = 10;
    Recombine Surface{TwoDimSurf};

    ids[] = Extrude {0, 0, 1}
    {
        Surface{TwoDimSurf};
        Layers{1};
        Recombine;
    };

    Physical Surface("outlet") = {ids[3]};
    Physical Surface("topAndBottom") = {ids[{2, 4}]};
    Physical Surface("inlet") = {ids[5]};
    Physical Surface("airfoil") = {ids[{6:8}]};
    Physical Surface("frontAndBack") = {ids[0], TwoDimSurf};
    Physical Volume("volume") = {ids[1]};""")


def write_box(alto, largo, mesh_dirname="mesh"):
    ancho = 0.5
    startpoint = 1000
    j = startpoint
    with open(os.path.join(mesh_dirname, "box.geo"), "w") as fid:

        fid.write("Point(%i) = {-%1.4f, -%1.4f, 0, %1.4f};\n" % (j, 0.4 * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {%1.4f, -%1.4f, 0, %1.4f};\n" % (j, (1 - 0.4) * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {%1.4f, %1.4f, 0, %1.4f};\n" % (j, (1 - 0.4) * largo, 0.5 * alto, ancho));
        j = j + 1
        fid.write("Point(%i) = {-%1.4f, %1.4f, 0, %1.4f};\n" % (j, 0.4 * largo, 0.5 * alto, ancho))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint, startpoint, startpoint + 1))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 1, startpoint + 1, startpoint + 2))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 2, startpoint + 2, startpoint + 3))
        fid.write("Line(%i) = {%i, %i};\n" % (startpoint + 3, startpoint + 3, startpoint))
        fid.write("Line Loop(3) = {%i, %i, %i, %i};\n" % tuple(range(startpoint, startpoint + 4)))


def write_perfil(coords, lc=0.002, mesh_dirname="mesh", bl=True):
    with open(os.path.join(mesh_dirname, "perfil.geo"), "w") as fid:
        fid.write("lc = %f;\n" % (lc))
        startpoint = 0
        endpoint = len(coords)-1
        midpoint = int(startpoint + 0.5 * (endpoint - startpoint))
        fid.write("perfilStartpoint = %i;\n" % startpoint)
        fid.write("perfilMidpoint = %i;\n" % midpoint)
        fid.write("perfilEndpoint = %i;\n" % endpoint)
        j = startpoint
        for x, y in coords:
            outputline = "Point(%i) = { %8.8f, %8.8f, 0.0, lc};\n" % (j, x, y)
            j = j + 1
            fid.write(outputline)

    if bl:

        coords_offset = offset_coords(coords)
        with open(os.path.join(mesh_dirname, "perfil_bl.geo"), "w") as fid:
            startpoint = j
            endpoint = startpoint+len(coords_offset)-1
            midpoint = int(startpoint + 0.5 * (endpoint - startpoint))
            fid.write("blStartpoint = %i;\n" % startpoint)
            fid.write("blMidpoint = %i;\n" % midpoint)
            fid.write("blEndpoint = %i;\n" % endpoint)
            for x, y in coords_offset:
                outputline = "Point(%i) = { %8.8f, %8.8f, 0.0, lc };\n" % (j, x, y)
                j = j + 1
                fid.write(outputline)


def get_kOmegaSTT_constants(Uref, intensity=0.01, length=1):
    """
    Returns k,omega for kOmegaSTT model. Input units should be consistent.
    :param Uref: Reference speed
    :param intensity: percentage of Uref
    :param length: reference length scale
    :return: k, omega
    """
    # from https://www.openfoam.com/documentation/guides/latest/doc/guide-turbulence-ras-k-omega-sst.html
    Cmu = 0.09
    k = 3/2 * (intensity*Uref)**2
    omega = k**(0.5) / Cmu / length
    return k, omega


def set_logger(verbose):
    level = {0: logging.WARN, 1: logging.INFO, 2: logging.DEBUG}[verbose]
    ch = logging.StreamHandler()
    ch.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    ch.setFormatter(formatter)
    logger.addHandler(ch)


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Runs simulation of airfoil geometry for a set of AoA")
    parser.add_argument("airfoil_filename", type=str, action="store", help="Airfoil coordinates file")
    parser.add_argument("basecase", type=str, action='store', help='basecase directory')
    parser.add_argument("aoa_list", type=float, nargs=3, default=(-15.0, 20.0, 1.0),
                        help="list of aoa: start, end, step")
    parser.add_argument("-o", "--output", type=str, action='store', required=True, help='destination directory')
    parser.add_argument("--with-bl", action="store_true", help="Use mesh with BL")
    parser.add_argument("--n-processors", type=int, default=0, choices=(0, 2, 4, 8),
                        help="Number of cores. By default 0, that mean, no paralelization is done")
    parser.add_argument('-d', '--docker-config', default=None, help='json file containing openfoams docker'
                                                             'container configuration')
    parser.add_argument('-v', '--verbose', action='count', default=0, help='be more verbose')
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_logger(args.verbose)
    filename = args.airfoil_filename
    basecase = args.basecase
    dirname = args.output
    mesh_dirname = os.path.join("meshes", "mesh_with_bl") if args.with_bl else os.path.join("meshes", "mesh")
    logger.debug('mesh dirname: %s', mesh_dirname)

    if args.docker_config:
        with open(args.docker_config, 'r') as fid:
            docker_config = json.load(fid)

    coords = read_aorfoil(filename)
    os.makedirs(dirname, exist_ok=True)
    aoa_list = np.arange(*args.aoa_list)
    df_coefficients_all = pd.DataFrame(columns=["aoa", "cl", "cd", "cm"])
    for aoa in aoa_list:
        destination_case = os.path.join(dirname, "case_aoa{:.1f}".format(aoa))
        try:
            if os.path.exists(destination_case):
                ans = "x"
                timeout = 10
                while ans.lower() not in "yn":
                    signal.alarm(timeout)
                    ans = input_with_timeout("el case {} ya existe. Desea volver a correrlo? [y, n]".format(destination_case))
                    signal.alarm(0)
                if ans == "y":
                    pass
                else:
                    continue
            else:
                destination_mesh_dirname = os.path.join(destination_case, mesh_dirname)
                logger.debug('destination mesh directory: %s', destination_mesh_dirname)

                logger.info("copy basecase to {}".format(destination_case))
                shutil.copytree(basecase, destination_case)

                logger.info("copy meshfiles to {}".format(destination_mesh_dirname))
                shutil.copytree(mesh_dirname, destination_mesh_dirname)

                logger.info("generate mesh for aoa {}".format(aoa))
                coords_rotated = apply_rotation(coords, aoa)
                write_box(8, 10, mesh_dirname=destination_mesh_dirname)
                write_perfil(coords_rotated, mesh_dirname=destination_mesh_dirname, bl=args.with_bl)
                command = "/usr/bin/gmsh -3 -o main.msh -format msh2 {}/main.geo".format(
                    mesh_dirname)
                logger.info("gmsh command: {}".format(command))
                subprocess.run(shlex.split(command), cwd=destination_case, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE,
                               universal_newlines=True)

            logger.info("Running case {}".format(destination_case))
            if args.docker_config:
                client = docker.from_env()
                try:
                    container_tag = docker_config['name']
                    container = client.containers.get(container_tag)
                except Exception as e:
                    logger.error(e)
                    logger.info('exiting script')
                    sys.exit(1)
                container.start()
                if args.n_processors > 0:
                    runfile = 'run_parallel.sh {}'.format(args.n_processors)
                else:
                    runfile = 'run.sh'

                logger.info("add bashrc command to runfile")
                write_bashrc_file(os.path.join(destination_case, runfile), docker_config["bashrc"])

                command = './{}'.format(runfile)
                logger.info('docker openfoam command: %s', command)
                exec_log = container.exec_run(shlex.split(command), stream=True, user='root', privileged=True,
                                              workdir=os.path.join(docker_config['home'], destination_case))
                for line in exec_log[1]:
                    logger.debug(line)
                container.stop()
            else:
                if args.n_processors > 0:
                    command = './run_parallel.sh {}'.format(args.n_processors)
                else:
                    command = './run.sh'
                logger.info('openfoam command: %s', command)
                subprocess.run(shlex.split(command), cwd=destination_case, stdout=subprocess.PIPE,
                               stderr=subprocess.PIPE, universal_newlines=True)

            logger.info("Calculations finished, getting results")
            df_coefficients = get_coefficients_df(destination_case)
            fig = plot_coefficients(df_coefficients)
            fig.savefig(os.path.join(destination_case, "aerodinamic_coefficients.png"))

            results = get_aerodynamic_coefficients(df_coefficients, method="fft", dirname=destination_case)
            results = results.append(pd.Series({"aoa": aoa}))
            logger.info("results = {}".format(results))
            df_coefficients_all = df_coefficients_all.append(results, ignore_index=True)

        except Exception as e:
            logger.exception(e)
            logger.info("Fail processing aoa {} with error:".format(aoa))

    df_coefficients_all.index = df_coefficients_all.aoa
    df_coefficients_all.drop("aoa", axis=1, inplace=True)

    # Update coefficient file
    output_coefficients_filename = os.path.join(dirname, "coefficients.txt")
    try:
        df_coefficients = pd.read_csv(output_coefficients_filename, index_col=1)
        df_coefficients = df_coefficients.astype('float64')
        logger.info("existing coefficients:")
        logger.info(df_coefficients)
        df_coefficients = df_coefficients.merge(df_coefficients_all, how="outer",  on=["aoa", "cl", "cd", "cm"])
        logger.info("updated coefficients")
        logger.info(df_coefficients)
    except FileNotFoundError:
        df_coefficients = df_coefficients_all

    with open(output_coefficients_filename, 'w') as fid:
        fid.write(df_coefficients_all.to_csv())

