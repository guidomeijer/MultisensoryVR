# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 16:11:21 2024 by Guido Meijer
"""

import numpy as np
import pandas as pd
import json
from os.path import join, isfile
from mayavi import mlab
from atlaselectrophysiology import rendering
from iblatlas import atlas
from one.webclient import http_download_file
from msvr_functions import paths, load_neural_data
from iblatlas.regions import BrainRegions
ba = atlas.AllenAtlas()
br = BrainRegions()
path_dict = paths()
url = 'http://download.alleninstitute.org/informatics-archive/current-release/mouse_ccf/annotation/ccf_2017/structure_meshes/'

# Settings
BRAIN_REGIONS = ['CA1', 'PERI', 'ECT', 'TEa', 'VISl']


# %% Download meshes 

for region in BRAIN_REGIONS:
    atlas_id = br.acronym2id(region)[0]
    if not isfile(join(path_dict['save_path'], str(atlas_id) + '.obj')):
        mesh_url = url + str(atlas_id) + '.obj'
        http_download_file(mesh_url, target_dir=path_dict['save_path'])

# %% Functions


def add_mesh(fig, obj_file, color=(1., 1., 1.), opacity=0.4):
    """
    Adds a mesh object from an *.obj file to the mayavi figure
    :param fig: mayavi figure
    :param obj_file: full path to a local *.obj file
    :param color: rgb tuple of floats between 0 and 1
    :param opacity: float between 0 and 1
    :return: vtk actor
    """

    import vtk

    reader = vtk.vtkOBJReader()
    reader.SetFileName(str(obj_file))
    reader.Update()
    mapper = vtk.vtkPolyDataMapper()
    mapper.SetInputConnection(reader.GetOutputPort())
    actor = vtk.vtkActor()
    actor.SetMapper(mapper)
    actor.GetProperty().SetOpacity(opacity)
    actor.GetProperty().SetColor(color)
    fig.scene.add_actor(actor)
    fig.scene.render()
    return mapper, actor


# %%

fig = rendering.figure(grid=False, size=(1024, 768))
for region in BRAIN_REGIONS:
    atlas_id = br.acronym2id(region)[0]
    mesh_path = join(path_dict['save_path'], f'{atlas_id}.obj')
    _, idx = br.id2index(atlas_id)
    color = br.rgb[idx[0][0], :] / 255
    add_mesh(fig, mesh_path, color, opacity=0.5)
    
rendering.rotating_video(join(path_dict['fig_path'], 'rotating_brain_regions.avi'),
                         fig, fps=30, secs=12)

    
    