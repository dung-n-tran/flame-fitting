import numpy as np
from os.path import join
from smpl_webuser.serialization import load_model
from fitting.util import write_simple_obj, safe_mkdir

if __name__ == "__main__":

    # Load FLAME model
    model_path = "./models/female_model.pkl"
    model = load_model(model_path)

    outmesh_dir = './output/helloworld'
    safe_mkdir(outmesh_dir)

    # Mean face
    outmesh_path = join(outmesh_dir, "meanface.obj")
    write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)

    # Sample a random pose
    model.pose[:] = np.random.randn(model.pose.size) * 0.05
    outmesh_path = join(outmesh_dir, "randompose.obj")
    write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)

    # Sample a random expression
    model.pose[:] = np.zeros(model.pose.size)
    model.betas[300:] = np.random.randn(model.betas[300:].size) * 1.0
    outmesh_path = join(outmesh_dir, "randomexpression.obj")
    write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)

    # Sample a random shape
    model.betas[:] = np.zeros(model.betas.size)
    model.betas[:300] = np.random.randn(model.betas[:300].size) * 1.0
    outmesh_path = join(outmesh_dir, "randomshape.obj")
    write_simple_obj(mesh_v=model.r, mesh_f=model.f, filepath=outmesh_path)
