#!/usr/bin/env python3

import argparse
import pygamer
import numpy as np
import pyvista as pv
import pyacvd
import vtk
from pathlib import Path
from lightdock.pdbutil.PDBIO import create_pdb_from_points


def parse_command_line():
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='surface-sampler')
    parser.add_argument("molecule", help="PDB file for input structure")
    parser.add_argument("distance", type=float, default=10.0, help="Distance to surface")
    parser.add_argument("points", type=int, default=400, help="Number of points to generate")
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_command_line()

    # Mesh the protein of interest
    # mesh = pygamer.readPDB_molsurf(args.molecule)
    mesh = pygamer.readPDB_gauss(args.molecule)

    # Compute the normal orientation
    components, orientable, manifold = mesh.compute_orientation()
    mesh.correctNormals()
    print(F"The mesh has {components} components, is"
          F" {'orientable' if orientable else 'non-orientable'}, and is"
          F" {'manifold' if manifold else 'non-manifold'}.")

    meshes = mesh.splitSurfaces()
    for i, m in enumerate(meshes):
        print(F"Mesh {i} is {m.getVolume()} A^3 in volume.")
    # Keep only the larger mesh
    mesh = meshes[0]

    for v in mesh.vertexIDs:
        v.data().selected = True
        # Apply 5 iterations of smoothing
    mesh.smooth(max_iter=5, preserve_ridges=False, verbose=True)

    for i in range(5):
        # Coarsen dense regions of the mesh
        mesh.coarse_dense(rate=2, numiter=3)
        # Coarsen flat regions of the mesh
        mesh.coarse_flat(rate=0.1, numiter=3)
        mesh.smooth(max_iter=3, preserve_ridges=True, verbose=False)
        print(F"Iteration {i+1}: {mesh.nVertices} vertices, {mesh.nEdges} edges, and {mesh.nFaces} faces.")

    # Center mesh at 0,0,0
    # center, radius = mesh.getCenterRadius()
    # mesh.translate(-center)

    # Set boundary markers of the mesh to 23
    for faceID in mesh.faceIDs:
        faceID.data().marker = 23

    # Get the root metadata
    gInfo = mesh.getRoot()
    gInfo.ishole = True    # Don't mesh the inside of
    gInfo.marker = -1

    path = Path(args.molecule)
    obj_name = f'{path.stem}.obj'
    pygamer.writeOBJ(obj_name, mesh)

    # Pyvista
    mesh = pv.read(obj_name)
    shell = mesh.decimate(0.97, volume_preservation=True).extract_surface()
    print(f'Decimation: {len(mesh.points)} -> {len(shell.points)}')

    # warp each point by the normal vectors
    for i in range(1,int(args.distance)+1):
        print(f'Expanding: {i}')
        shell = shell.compute_normals()
        warp = vtk.vtkWarpVector()
        warp.SetInputData(shell)
        warp.SetInputArrayToProcess(0, 0, 0,
                                    vtk.vtkDataObject.FIELD_ASSOCIATION_POINTS,
                                    vtk.vtkDataSetAttributes.NORMALS)
        warp.SetScaleFactor(2)
        warp.Update()
        shell = pv.wrap(warp.GetOutput())

    expanded_mesh = shell.extract_surface()
    clus = pyacvd.Clustering(expanded_mesh)
    clus.subdivide(3)
    clus.cluster(args.points)
    shell = clus.create_mesh().extract_surface()

    uniform = shell
    p = pv.Plotter(notebook=False, shape=(1,1))
    p.add_mesh(shell)
    p.add_points(np.asarray(uniform.points), color="r",
                 point_size=8.0, render_points_as_spheres=True)
    p.add_mesh(expanded_mesh, smooth_shading=True)
    p.link_views()
    p.show_bounds()
    p.show()

    create_pdb_from_points(f'{path.stem}_swarms.pdb', uniform.points)
