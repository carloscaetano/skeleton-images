import KinectData as kd
import numpy as np
import cv2
import os
import math
from typing import Type, Dict, Union, List
from KinectData import KinectData

class ImgType(object):
    kinect_data: KinectData

    #NTU 25 joints
    depth_first_traversal_skl_NTU = [1, 20, 2, 3, 2, 20, 4, 5, 6, 7, 21, 22, 21, 7, 6, 5, 4, 20,
                                     8, 9, 10, 11, 23, 24, 23, 11, 10, 9, 8, 20, 1, 0, 12, 13,
                                     14, 15, 14, 13, 12, 0, 16, 17, 18, 19, 18, 17, 16, 0, 1]

    def __init__(self) -> None:
        self.temporal_scale = [1]
        self.kinect_data = kd.KinectData()
        self.width = 0
        self.height = 0
        self.channels = 0
        self.img_list = []
        self.img_type = ''
        self.num_imgs = 0

    def __init__(self, img_type: str) -> None:
        self.temporal_scale = [1]
        self.kinect_data = kd.KinectData()
        self.width = 0
        self.height = 0
        self.channels = 0
        self.img_list = []
        self.img_type = img_type
        self.num_imgs = 0

    def __del__(self) -> None:
        del self.img_list
        del self.kinect_data

    def compute_joint_difference(self, i_frame: int, j_joint: int, k_body: int) -> (float, float, float):
        ret = (0.0, 0.0, 0.0)
        if (i_frame+1) < self.kinect_data.n_frames and self.kinect_data.kinect_blocks[i_frame+1].n_bodies > k_body:
            joint_data_f1 = self.kinect_data.kinect_blocks[i_frame].body_list[k_body].joint_data[j_joint]
            joint_data_f2 = self.kinect_data.kinect_blocks[i_frame+1].body_list[k_body].joint_data[j_joint]
            diff_x = joint_data_f1.x_joint - joint_data_f2.x_joint
            diff_y = joint_data_f1.y_joint - joint_data_f2.y_joint
            diff_z = joint_data_f1.z_joint - joint_data_f2.z_joint
            ret = (diff_x, diff_y, diff_z)
        return ret

    def generate_file_name(self, skl_file: str, k_body: int) -> str:
        f_name = os.path.basename(skl_file)
        file_prefix = f_name.split('.')[0] + '_'
        file_suffix = '.' + f_name.split('.')[1]
        f_save = file_prefix + str(k_body + 1) + '_' + self.img_type + file_suffix
        return f_save

    def save_img_list(self, list_path_to_save: List[str]) -> None:
        for i in range(len(self.img_list)):
            np.savez(list_path_to_save[i], self.img_list[i])

    def compute_temporal_joint_difference(self, i_frame: int, j_joint: int, k_body: int, temporal_dist: int) -> (float, float, float):
        ret = (0.0, 0.0, 0.0)
        if (i_frame + temporal_dist) < self.kinect_data.n_frames and self.kinect_data.kinect_blocks[i_frame + temporal_dist].n_bodies > k_body:
            joint_data_f1 = self.kinect_data.kinect_blocks[i_frame].body_list[k_body].joint_data[j_joint]
            joint_data_f2 = self.kinect_data.kinect_blocks[i_frame + temporal_dist].body_list[k_body].joint_data[j_joint]
            diff_x = joint_data_f1.x_joint - joint_data_f2.x_joint
            diff_y = joint_data_f1.y_joint - joint_data_f2.y_joint
            diff_z = joint_data_f1.z_joint - joint_data_f2.z_joint
            ret = (diff_x, diff_y, diff_z)
        return ret

    @staticmethod
    def normalize(value: float, lower_bound: float, higher_bound: float, max_value: int, min_value: int) -> float:
        if value > higher_bound:
            ret_value = max_value
        elif value < lower_bound:
            ret_value = min_value
        else:
            ret_value = (max_value * ((value - lower_bound) / (higher_bound - lower_bound)))  # estava com cast de int()
        return ret_value

    @staticmethod
    def convert_to_uint8(img: np.ndarray, lower_bound: float, higher_bound: float) -> np.ndarray:
        img_return = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_return[i, j] = (ImgType.normalize(img[i, j][0], lower_bound, higher_bound, 255, 0),
                                    ImgType.normalize(img[i, j][1], lower_bound, higher_bound, 255, 0),
                                    ImgType.normalize(img[i, j][2], lower_bound, higher_bound, 255, 0))
        return img_return

    @staticmethod
    def convert_to_uint8_ntu_skl(img: np.ndarray) -> np.ndarray:
        '''convert to uint image based on higher and lower values from NTU training skeletons'''
        img_return = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_return[i, j] = (ImgType.normalize(img[i, j][0], -3.0, 3.0, 255, 0),
                                    ImgType.normalize(img[i, j][1], -2.0, 2, 255, 0),
                                    ImgType.normalize(img[i, j][2], 0.5, 4.0, 255, 0))
        return img_return

    @staticmethod
    def convert_to_uint8_ntu_diff_skl(img: np.ndarray) -> np.ndarray:
        '''convert to uint image based on higher and lower values from NTU training skeleton differences'''
        img_return = np.zeros(img.shape, np.uint8)
        for i in range(img.shape[0]):
            for j in range(img.shape[1]):
                img_return[i, j] = (ImgType.normalize(img[i, j][0], -3.36, 2.23, 255, 0),
                                    ImgType.normalize(img[i, j][1], -1.02, 1.10, 255, 0),
                                    ImgType.normalize(img[i, j][2], -4.34, 3.17, 255, 0))
        return img_return

    @staticmethod
    def show_skl_img(img: np.ndarray, str_text: str = 'Image') -> None:
        cv2.imshow(str_text, img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        print(str_text)

    @staticmethod
    def show_skl_img_colormap(img: np.ndarray, str_text: str = 'Image') -> None:
        img_u = ImgType.convert_to_uint8(img, 0.0, 1.0)
        #cv2.imshow(str_text, cv2.applyColorMap(img_u, cv2.COLORMAP_JET))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        cv2.imwrite(str_text, cv2.applyColorMap(img_u, cv2.COLORMAP_HSV))
        print(str_text)

    def set_width(self, width: int) -> None:
        self.width = width

    def set_height(self, height: int) -> None:
        self.height = height

    def set_channels(self, channels: int) -> None:
        self.channels = channels

    def set_temporal_scale(self, temporal_scale: List) -> None:
        self.temporal_scale = temporal_scale


class CaetanoMagnitude(ImgType):
    def __init__(self) -> None:
        super().__init__('CaetanoMagnitude')
        self.width_resized = 100
        self.height_resized = len(self.depth_first_traversal_skl_NTU)  # 25
        self.num_imgs = 1
        self.channels = len(self.temporal_scale)

    @staticmethod
    def compute_joint_magnitude(diff_joint: np.array) -> float:
        ret = (diff_joint ** 2).sum() ** (1. / 2)
        ret = ImgType.normalize(ret, 0.0, 1, 1.0, 0.0)
        return ret

    def process_skl_file(self, skl_file: str, path_to_save: str) -> str:
        extraction = ''
        try:
            self.height_resized = len(ImgType.depth_first_traversal_skl_NTU)
            if len(self.temporal_scale) > 1:
                self.set_channels(len(self.temporal_scale))  # varies according to method
            else:
                self.set_channels(3)
            self.kinect_data.read_data(skl_file)
            self.set_height(len(ImgType.depth_first_traversal_skl_NTU))  # self.set_height(self.kinect_data.n_joints)
            self.set_width(self.kinect_data.n_frames)
            list_path_to_save = []
            for k_body in range(self.kinect_data.n_bodies):
                img = np.zeros((self.height, self.width, self.channels), np.float)
                for i_frames in range(self.kinect_data.n_frames):
                    if self.kinect_data.kinect_blocks[i_frames].n_bodies > k_body:
                        for j_pos in range(len(ImgType.depth_first_traversal_skl_NTU)):
                            j_joints = ImgType.depth_first_traversal_skl_NTU[j_pos]
                            # TODO: version with normalization by the neck
                            mag_values = []
                            for t_scale in self.temporal_scale:
                                diff_joint = np.array(self.compute_temporal_joint_difference(i_frames, j_joints, k_body, t_scale))
                                mag_values.append(self.compute_joint_magnitude(diff_joint))
                            img[j_pos, i_frames] = tuple(mag_values)
                self.img_list.append(cv2.resize(img, (self.width_resized, self.height_resized)))
                f_save = self.generate_file_name(skl_file, k_body)
                list_path_to_save.append(os.path.join(path_to_save, f_save))
            self.save_img_list(list_path_to_save)
            extraction = 'OK\t' + skl_file + '\t' + str(self.temporal_scale)
            print(extraction)
        except Exception as exception:
            extraction = 'ERROR\t' + skl_file + ' ' + str(exception)
            print(extraction)
        finally:
            del list_path_to_save
            self.img_list.clear()
            return extraction


class CaetanoOrientation(ImgType):
    x: int = 0
    y: int = 1
    z: int = 2
    filter_by_magnitude = True
    diff_ori_val = 1.1
    mag_threshold = 0.004

    def __init__(self) -> None:
        super().__init__('CaetanoOrientation')
        self.width_resized = 100
        self.height_resized = len(self.depth_first_traversal_skl_NTU)  # 25
        self.num_imgs = 1
        self.channels = len(self.temporal_scale)

    def compute_joint_orientation(self, diff_joint: np.array, fir_axis: int, sec_axis: int) -> float:
        ret = math.atan2(diff_joint[fir_axis], diff_joint[sec_axis]) * 180 / math.pi
        ret = self.normalize(ret, -180.0, 180.0, 1.0, -1.0)
        return ret

    def process_skl_file(self, skl_file: str, path_to_save: str) -> str:
        extraction = ''
        try:
            self.height_resized = len(ImgType.depth_first_traversal_skl_NTU)
            if len(self.temporal_scale) > 1:
                self.set_channels(len(self.temporal_scale) * 3)  # varies according to method
            else:
                self.set_channels(3)
            self.kinect_data.read_data(skl_file)
            self.set_height(len(ImgType.depth_first_traversal_skl_NTU))  # self.set_height(self.kinect_data.n_joints)
            self.set_width(self.kinect_data.n_frames)
            list_path_to_save = []
            for k_body in range(self.kinect_data.n_bodies):
                img = np.zeros((self.height, self.width, self.channels), np.float)
                for i_frames in range(self.kinect_data.n_frames):
                    if self.kinect_data.kinect_blocks[i_frames].n_bodies > k_body:
                        for j_pos in range(len(ImgType.depth_first_traversal_skl_NTU)):
                            j_joints = ImgType.depth_first_traversal_skl_NTU[j_pos]
                            # TODO: version with normalization by the neck
                            ori_values = []
                            for t_scale in self.temporal_scale:
                                diff_joint = np.array(self.compute_temporal_joint_difference(i_frames, j_joints, k_body, t_scale))

                                if CaetanoOrientation.filter_by_magnitude:
                                    mag_val = CaetanoMagnitude.compute_joint_magnitude(diff_joint)
                                    if mag_val < CaetanoOrientation.mag_threshold * t_scale:
                                        ori_yx_value = CaetanoOrientation.diff_ori_val
                                        ori_yz_value = CaetanoOrientation.diff_ori_val
                                        ori_zx_value = CaetanoOrientation.diff_ori_val
                                    else:
                                        ori_yx_value = self.compute_joint_orientation(diff_joint, self.y, self.x)
                                        ori_yz_value = self.compute_joint_orientation(diff_joint, self.y, self.z)
                                        ori_zx_value = self.compute_joint_orientation(diff_joint, self.z, self.x)
                                else:
                                    ori_yx_value = self.compute_joint_orientation(diff_joint, self.y, self.x)
                                    ori_yz_value = self.compute_joint_orientation(diff_joint, self.y, self.z)
                                    ori_zx_value = self.compute_joint_orientation(diff_joint, self.z, self.x)

                                ori_values.append(ori_yx_value)
                                ori_values.append(ori_yz_value)
                                ori_values.append(ori_zx_value)
                            img[j_pos, i_frames] = tuple(ori_values)
                self.img_list.append(cv2.resize(img, (self.width_resized, self.height_resized)))
                f_save = self.generate_file_name(skl_file, k_body)
                list_path_to_save.append(os.path.join(path_to_save, f_save))
            self.save_img_list(list_path_to_save)
            extraction = 'OK\t' + skl_file + '\t' + str(self.temporal_scale)
            print(extraction)
        except Exception as exception:
            extraction = 'ERROR\t' + skl_file + ' ' + str(exception)
            print(extraction)
        finally:
            del list_path_to_save
            self.img_list.clear()
            return extraction


class_img_types: Dict[int, Type[Union[CaetanoMagnitude, CaetanoOrientation]]] = {
    1: CaetanoMagnitude,
    2: CaetanoOrientation
}

