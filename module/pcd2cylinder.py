import numpy as np
import math

#点转扇形网格

def points_to_cylinder_dynamic(points, #
                        # [3.6, 0.5, 8] 表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
                       lider_range=[360,50,8], #一个值，点距离激光雷达的最大水平距离
                       cylinder_shape=[101,101,1],
                       max_points=100, #扇形网格中点的最大数量
                       max_cylinder=8000):
# define cylinder_shape first : 128,64,1,
    cylinder_shape = tuple(np.round(cylinder_shape).astype(np.int32).tolist()) # 100x100x1
    num_points_per_cylinder = np.zeros(shape=(max_cylinder, ), dtype=np.int32) # 80000
    coor_to_cylinderidx = -np.ones(shape=cylinder_shape, dtype=np.int32) # 100x100
    cylinder = np.zeros(
        shape=(max_cylinder, max_points, points.shape[-1]+1), dtype=points.dtype) #8000*100*5
    coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32) #8000*3
    cylinder_num = 0
    N = points.shape[0] # Number of points in a frame

    cylinder_size = [3.6, 0, 8]
    cylinder_size[1]=[]
    cylinder_size[1].append(0)
    start_r=0.1
    k_r=0.005

    for j in range(1, 101):
        temp= start_r+ k_r*j
        temp += cylinder_size[1][j-1]
        cylinder_size[1].append(temp)

    for i in range(N):
        coor = np.zeros(shape=(3,), dtype=np.int32)
        # segid=np.floor(math.atan2(points[i,1],points[i,0]) + math.pi/(voxel_size[0]/180*math.pi)) #????
        rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2) # 点到圆心距离
        for dis in cylinder_size[1]:
            if rad_val-dis<0:
                rad_id=cylinder_size[1].index(dis) # 这个点分配到哪个bin
                break
        ang_id = np.floor((math.atan2(points[i, 1], points[i, 0]) / math.pi * (
                    180 / cylinder_size[0])) + 50)  # 起始点从第三象限开始逆时针旋转(0->99)
        height_id=np.floor((points[i,2]+cylinder_size[2]/2)/cylinder_size[2])

        if ang_id > 100:
            print(str(i)+ ": "+str(ang_id)+ "ang_id out of range")
        elif rad_id > 100:
            print(str(i)+": "+ str(rad_id) +"rad_id out of range")

        coor[0]=ang_id
        coor[1]=rad_id
        coor[2]=height_id
        #需要重新命名
        cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]] # 100x100x1

        if cylinderidx==-1:
            cylinderidx = cylinder_num
            if cylinder_num >= max_cylinder:
                break
            cylinder_num+=1
            coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx #记录个数在对应位置（一次）
            coors[cylinderidx] = coor # 记录cylinder坐标到coors(一次)
        num = num_points_per_cylinder[cylinderidx]  # 与coor_to_cylinderidx 功能一样

        if num < max_points:
            point= points[i].tolist()
            point.append(rad_val)
            cylinder[cylinderidx, num] = point
            num_points_per_cylinder[cylinderidx] += 1

    pts_in_cylinder = cylinder[:cylinder_num]  # p x n x 4+1
    cylinder_coors=coors[:cylinder_num] # p x 3
    num_points_per_cylinder = num_points_per_cylinder[:cylinder_num] # p x 1
    return pts_in_cylinder, cylinder_coors, num_points_per_cylinder  # 共同点是 p


def points_to_cylinder_fixed(points, #
                        # r= 40 #一个值，点距离激光雷达的最大水平距离
                       cylinder_size=[3.6,0.5,8], #表示扇形旋角度为3.6度，扇形水平距离为1米，高度为8米
                       cylinder_shape=[100,100,1],
                       max_points=100, #扇形网格中点的最大数量
                       max_cylinder=8000):
# define cylinder_shape first : 128,64,1,

    cylinder_shape = tuple(np.round(cylinder_shape).astype(np.int32).tolist()) # 100x100x1
    num_points_per_cylinder = np.zeros(shape=(max_cylinder, ), dtype=np.int32) # 80000
    coor_to_cylinderidx = -np.ones(shape=cylinder_shape, dtype=np.int32) # 100x100
    cylinder = np.zeros(
        shape=(max_cylinder, max_points, points.shape[-1]+1), dtype=points.dtype) #8000*100*5
    coors = np.zeros(shape=(max_cylinder, 3), dtype=np.int32) #8000*3
    cylinder_num = 0
    N = points.shape[0] # Number of points in a frame
    cylinder_size =cylinder_size


    for i in range(N):
        coor = np.zeros(shape=(3,), dtype=np.int32)
        rad_val = math.sqrt(points[i, 0] ** 2 + points[i, 1] ** 2) # 点到圆心距离
        # rad_val= np.linalg.norm([points[i,0],points[i,1]])
        rad_id= rad_val/ cylinder_size[1]
        ang_id = np.floor((math.atan2(points[i, 1], points[i, 0]) / math.pi * (
                    180 / cylinder_size[0])) + 50)  # 起始点从第三象限开始逆时针旋转(0->99)
        height_id=np.floor((points[i,2]+cylinder_size[2]/2)/cylinder_size[2])

        if ang_id >= 100:
            print(str(i)+ "ang_id out of range")
        elif rad_id >= 100:
            print(str(i)+"rad_id out of range")

        coor[0]=ang_id
        coor[1]=rad_id
        coor[2]=height_id
        #需要重新命名
        cylinderidx = coor_to_cylinderidx[coor[0], coor[1], coor[2]] # 100x100x1

        if cylinderidx==-1:
            cylinderidx = cylinder_num
            if cylinder_num >= max_cylinder:
                break
            cylinder_num+=1
            coor_to_cylinderidx[coor[0], coor[1], coor[2]] = cylinderidx #记录个数在对应位置（一次）
            coors[cylinderidx] = coor # 记录cylinder坐标到coors(一次)
        num = num_points_per_cylinder[cylinderidx]  # 与coor_to_cylinderidx 功能一样

        if num < max_points:
            point= points[i].tolist()
            point.append(rad_val)
            cylinder[cylinderidx, num] = point
            num_points_per_cylinder[cylinderidx] += 1

    pts_in_cylinder = cylinder[:cylinder_num]  # p x n x 4+1
    cylinder_coors=coors[:cylinder_num] # p x 3
    num_points_per_cylinder = num_points_per_cylinder[:cylinder_num] # p x 1
    return pts_in_cylinder, cylinder_coors, num_points_per_cylinder  # 共同点是 p