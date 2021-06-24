# play motion of pcd visualization
import open3d as o3d
import os
import time

# Play pcds
def play_motion(pcd_path: []):
    play_motion.vis = o3d.visualization.Visualizer()
    play_motion.index = 0

    def forward(vis):
        pm = play_motion
        if pm.index < len(pcd_path) - 1:
            pm.index += 1
            cloud = o3d.io.read_point_cloud(pcd_path[pm.index])
            pcd.points = cloud.points
            pcd.colors = cloud.colors
            time.sleep(.2)
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()
        return False

    # Geometry of the initial frame
    pcd = o3d.geometry.PointCloud()
    cloud = o3d.io.read_point_cloud(pcd_path[0])
    pcd.points = cloud.points
    pcd.colors = cloud.colors
    # pcd.colors = o3d.utility.Vector3dVector(np.ones(cloud.shape) * [1,0,0])

    # Initialize Visualizer and start animation callback
    vis = play_motion.vis
    vis.create_window()
    vis.add_geometry(pcd)
    vis.register_animation_callback(forward)
    vis.run()
    vis.destroy_window()


root = "/velo_pcd/"
part = os.listdir(root)
data_path = []
for idx in part:
    data_path.append(os.path.join(root, idx))

play_motion(data_path)
