import open3d as o3d
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import pdist, cdist

def get_alpha(loc,subset=200,k=200):
    dis = cdist(loc,loc[np.random.permutation(loc.shape[0])[:subset]])
    return np.median(np.sort(dis,axis=0)[200])

def smooth_mesh(mesh,taubin_iter=None,subdivide_iter=3,show=False):
    """Smooth Mesh.

    Smoothing a mesh for a tissue. 
    
    Args:
        taubin_iter: A `Float` value indicating the number of iterations for taubin smooth.
        subdivide_iter:  A `Float` value indicating the number of iterations for subdivide smooth.
        show: A `Boolean` value indicating whether to show the mesh.

    Returns:
        A smoothed mesh object.
    """
    if taubin_iter is not None:
        print(f'filter with Taubin with {taubin_iter} iterations')
        mesh = mesh.filter_smooth_taubin(number_of_iterations=taubin_iter)
        mesh.compute_vertex_normals()
    if subdivide_iter is not None:
        mesh = mesh.subdivide_loop(number_of_iterations=subdivide_iter)
        print(
            f'After subdivision it has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles'
        )
    if show:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)
    return mesh

def create_mesh(loc,alpha=None,show=False):
    """Create Mesh.

    Creating a mesh for a tissue. 
    
    Args:
        loc: A `DataFrame` object represents the 3D location of each spot.
        alpha: A `Float` value used for o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape function. A large alpha, will ignore more details.
        show: A `Boolean` value indicating whether to show the mesh.

    Returns:
        A mesh object.
    """
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(loc)
    if alpha is None:
        alpha = get_alpha(loc)
    print(f"alpha={alpha:.3f}")
    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_alpha_shape(pcd, alpha)
    mesh.compute_vertex_normals()
    if show:
        o3d.visualization.draw_geometries([pcd,mesh], mesh_show_back_face=True)
    return mesh

def sample_in_mesh(mesh,xyz,num_sample=500000,save_sampled=None,save_surface=None):
    """Sample spots/cells in a mesh.

    Sampling spots/cells in a mesh.
    
    Args:
        mesh: A mesh object to be sampled.
        xyz:  A matrix of x, y, and z coordinates of each spot/cell.
        num_sample: A `int` value indicating the number of spots/cells to be sampled.
        save_surface: A `Boolean` value indicating whether to save the spots/cells on the surface of the mesh.

    Returns:
        A smoothed mesh object.
    """
    mesh_scene = o3d.t.geometry.TriangleMesh.from_legacy(mesh)

    # Create a scene and add the triangle mesh
    scene = o3d.t.geometry.RaycastingScene()
    _ = scene.add_triangles(mesh_scene)

    dx,dy,dz = xyz.max(0)-xyz.min(0)
    dd = np.math.pow(num_sample/((dx/dz)*(dy/dz)*(dz/dz)), 1/3)

    nx,ny,nz=int((dx/dz)*dd),int((dy/dz)*dd),int(dd)
    xmin,ymin,zmin=xyz.min(0)
    xmax,ymax,zmax=xyz.max(0)
    x_sampled = np.linspace(xmin-1e-7, xmax+1e-7, nx)
    y_sampled = np.linspace(ymin-1e-7, ymax+1e-7, ny)
    z_sampled = np.linspace(zmin-1e-7, zmax+1e-7, nz)

    xyz_sampled = []
    for x_s in x_sampled:
        for y_s in y_sampled:
            for z_s in z_sampled:
                xyz_sampled.append([x_s,y_s,z_s])

    xyz_sampled = np.array(xyz_sampled)

    query_point = o3d.core.Tensor(xyz_sampled, dtype=o3d.core.Dtype.Float32)

    # Compute distance of the query point from the surface
    unsigned_distance = scene.compute_distance(query_point)
    signed_distance = scene.compute_signed_distance(query_point)
    occupancy = scene.compute_occupancy(query_point)

    occupancy = np.array(occupancy)

    xyz_sampled_inmesh = np.array(xyz_sampled)[occupancy == 1]
    xyz_surface = np.asarray(mesh.vertices)
    if save_sampled is not None:
        np.save(save_sampled,xyz_sampled_inmesh)
    if save_surface is not None:
        np.save(save_surface,xyz_surface)
    return xyz_sampled_inmesh, xyz_surface

def convert_colors(val, cmap='Spectral_r'):
    val = (val - val.min())/(val.max()-val.min())
    cmap = plt.get_cmap(cmap)
    # point_colors = [cmap(val) for val in ano4_gpr]
    point_colors = [cmap(v) for v in val]
    point_colors = np.array(point_colors)[:,:3]
    return point_colors

def get_surface_colors(mesh, surface_expr, cmap='Spectral_r'):
    """Obtaining the color of the surface of mesh.

    Obtaining the color of the surface of mesh according to the expression value.
    
    Args:
        mesh: A mesh object to be colored.
        surface_expr:  A expression matrix of each spots/cells at the surface of mesh.
        cmap: The cmap used for generating colors.
        
    Returns:
        ``None``
    """
    surface_expr = (surface_expr - surface_expr.min())/(surface_expr.max()-surface_expr.min())
    cmap = plt.get_cmap(cmap)
    point_colors = [cmap(val) for val in surface_expr]
    point_colors = np.array(point_colors)[:,:3]
    mesh.vertex_colors = o3d.utility.Vector3dVector(point_colors)
    
def save_view_parameters(mesh, filename, point_size=3):
    """Saving the view parameters.

    Saving the view parameters according to user's opperations.
    
    Args:
        mesh: A mesh object to be shown.
        filename: The path where the parameters will be saved.
        point_size: The size of points to be shown.
        
    Returns:
        ``None``
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size=point_size
    vis.add_geometry(mesh)
    vis.run()
    param = vis.get_view_control().convert_to_pinhole_camera_parameters()
    o3d.io.write_pinhole_camera_parameters(filename,param)
    vis.destroy_window()
    
def load_view_parameters(mesh, filename, point_size=3):
    """Loading the view parameters.

    Loading the view parameters from a file.
    
    Args:
        mesh: A mesh object to be shown.
        filename: The path where the parameters saved.
        point_size: The size of points to be shown.
        
    Returns:
        ``None``
    """
    vis = o3d.visualization.Visualizer()
    vis.create_window()
    ctr = vis.get_view_control()
    opt = vis.get_render_option()
    opt.mesh_show_back_face = True
    opt.point_size=point_size
    param = o3d.io.read_pinhole_camera_parameters(filename)
    vis.add_geometry(mesh)
    ctr.convert_from_pinhole_camera_parameters(param)
    vis.run()
    vis.destroy_window()