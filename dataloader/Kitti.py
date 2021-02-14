import cv2
import numpy as np

def __velo_2_img_projection(self, calib ,points):
    """ convert velodyne coordinates to camera image coordinates """

    for line in self.c2c_file:
        (key, val) = line.split(':', 1)
        if key == ('P_rect_' + mode):
            P_ = np.fromstring(val, sep=' ')
            P_ = P_.reshape(3, 4)
            # erase 4th column ([0,0,0])
            P_ = P_[:3, :3]
    # rough velodyne azimuth range corresponding to camera horizontal fov
    if self.__h_fov is None:
        self.__h_fov = (-50, 50)
    if self.__h_fov[0] < -50:
        self.__h_fov = (-50,) + self.__h_fov[1:]
    if self.__h_fov[1] > 50:
        self.__h_fov = self.__h_fov[:1] + (50,)

    # R_vc = Rotation matrix ( velodyne -> camera )
    # T_vc = Translation matrix ( velodyne -> camera )
    R_vc, T_vc = self.__calib_velo2cam()

    # P_ = Projection matrix ( camera coordinates 3d points -> image plane 2d points )
    P_ = self.__calib_cam2cam()

    """
    xyz_v - 3D velodyne points corresponding to h, v FOV limit in the velodyne coordinates
    c_    - color value(HSV's Hue vaule) corresponding to distance(m)
             [x_1 , x_2 , .. ]
    xyz_v =  [y_1 , y_2 , .. ]
             [z_1 , z_2 , .. ]
             [ 1  ,  1  , .. ]
    """
    xyz_v, c_ = self.__point_matrix(points)

    """
    RT_ - rotation matrix & translation matrix
        ( velodyne coordinates -> camera coordinates )
            [r_11 , r_12 , r_13 , t_x ]
    RT_  =  [r_21 , r_22 , r_23 , t_y ]
            [r_31 , r_32 , r_33 , t_z ]
    """
    RT_ = np.concatenate((R_vc, T_vc), axis=1)

    # convert velodyne coordinates(X_v, Y_v, Z_v) to camera coordinates(X_c, Y_c, Z_c)
    for i in range(xyz_v.shape[1]):
        xyz_v[:3, i] = np.matmul(RT_, xyz_v[:, i])

    """
    xyz_c - 3D velodyne points corresponding to h, v FOV in the camera coordinates
             [x_1 , x_2 , .. ]
    xyz_c =  [y_1 , y_2 , .. ]
             [z_1 , z_2 , .. ]
    """
    xyz_c = np.delete(xyz_v, 3, axis=0)

    # convert camera coordinates(X_c, Y_c, Z_c) image(pixel) coordinates(x,y)
    for i in range(xyz_c.shape[1]):
        xyz_c[:, i] = np.matmul(P_, xyz_c[:, i])

    """
    xy_i - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates before scale adjustment
    ans  - 3D velodyne points corresponding to h, v FOV in the image(pixel) coordinates
             [s_1*x_1 , s_2*x_2 , .. ]
    xy_i =   [s_1*y_1 , s_2*y_2 , .. ]        ans =   [x_1 , x_2 , .. ]
             [  s_1   ,   s_2   , .. ]                [y_1 , y_2 , .. ]
    """
    xy_i = xyz_c[::] / xyz_c[::][2]
    ans = np.delete(xy_i, 2, axis=0)

    return ans, c_


def velo_projection_frame(self, h_fov=None, v_fov=None, x_range=None, y_range=None, z_range=None):
    """ print velodyne 3D points corresponding to camera 2D image """

    self.__v_fov, self.__h_fov = v_fov, h_fov
    self.__x_range, self.__y_range, self.__z_range = x_range, y_range, z_range
    velo_gen, cam_gen = self.velo_file, self.camera_file

    if velo_gen is None:
        raise ValueError("Velo data is not included in this class")
    if cam_gen is None:
        raise ValueError("Cam data is not included in this class")
    res, c_ = self.__velo_2_img_projection(velo_gen)
    return cam_gen, res, c_
def read_calib(file):
    calib_dict ={}
    with open(file,'r') as f:
        for x in f:
          key,val = x.split(':')
          if key == 'P0':
            P_ = P_.reshape(3, 4)
            P_ = P_[:3, :3]
          if key == 'R':
              R = np.fromstring(val, sep=' ')
              R = R.reshape(3, 3)
          if key == 'T':
              T = np.fromstring(val, sep=' ')
              T = T.reshape(3, 1)
        return P_,R,T


if __name__ == "__main__":

    im = cv2.imread()
    v_fov, h_fov = (-24.9, 2.0), (-90, 90)
    calib_root = '/mnt/AAB281B7B2818911/Thesis/implementations'
    v2c_filepath = './calib_velo_to_cam.txt'
    c2c_filepath = './calib_cam_to_cam.txt'
    points = np.fromfile(os.path.join(calib_root,), dtype=np.float32).reshape(-1, 4)
