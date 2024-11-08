# -*- coding: utf-8 -*-
# Author: Runsheng Xu <rxx3386@ucla.edu>
# License: TDG-Attribution-NonCommercial-NoDistrib

"""
Dataset class for intermediate fusion
"""
import random
import math
import warnings
from collections import OrderedDict

import numpy as np
import torch

import opencood.data_utils.datasets
import opencood.data_utils.post_processor as post_processor
from opencood.utils import box_utils
from opencood.data_utils.datasets import online_basedataset
from opencood.data_utils.pre_processor import build_preprocessor
from opencood.utils.pcd_utils import \
    mask_points_by_range, mask_ego_points, shuffle_points, \
    downsample_lidar_minimum
from opencood.utils.transformation_utils import x1_to_x2


class OnlineIntermediateFusionDataset(online_basedataset.OnlineBaseDataset):
    """
    This class is for intermediate fusion where each vehicle transmit the
    deep features to ego.
    """

    def __init__(self, params, visualize, train=True):
        super(OnlineIntermediateFusionDataset, self). \
            __init__(params, visualize, train)

        # if project first, cav's lidar will first be projected to
        # the ego's coordinate frame. otherwise, the feature will be
        # projected instead.
        self.proj_first = True
        if 'proj_first' in params['fusion']['args'] and \
                not params['fusion']['args']['proj_first']:
            self.proj_first = False

        # whether there is a time delay between the time that cav project
        # lidar to ego and the ego receive the delivered feature
        self.cur_ego_pose_flag = True if 'cur_ego_pose_flag' not in \
                                         params['fusion']['args'] else \
            params['fusion']['args']['cur_ego_pose_flag']

        self.pre_processor = build_preprocessor(params['preprocess'],
                                                train)
        self.post_processor = post_processor.build_postprocessor(
            params['postprocess'],
            train)

    def data_reformat(self, observations):
        """
        Receive data from Scene Generator and convert the right format
        for cooperative perception models.

        Parameters
        ----------
        observations : dict
            The dictionary that contains all cavs' info including lidar pose
            and lidar observations.

        gt_info : dict
            groundtruth information for all objects in the scene (not
            neccessary all in the valid range).

        Returns
        -------
        A dictionary that contains the data with the right format that
        detection model needs.
        """

        processed_data_dict = OrderedDict()
        processed_data_dict['ego'] = {}

        assert len(observations) == 1, "Only single agents are allowed"

        ego_lidar_pose = observations[0]['params']["ego_pose"]

        processed_features = []

        # prior knowledge for time delay correction and indicating data type
        # (V2V vs V2i)
        velocity = []
        time_delay = []
        infra = []
        spatial_correction_matrix = []

        # loop over all CAVs to process information
        for cav_id, selected_cav_base in observations.items():
            # check if the cav is within the communication range with ego
            selected_cav_lidar_pose = selected_cav_base['params']['lidar_pose']
            distance = \
                math.sqrt(
                    (selected_cav_lidar_pose[0] - ego_lidar_pose[0]) ** 2 +
                    (selected_cav_lidar_pose[1] - ego_lidar_pose[1]) ** 2)

            if distance > opencood.data_utils.datasets.COM_RANGE:
                continue

            selected_cav_processed, void_lidar = self.get_item_single_car(
                selected_cav_base,
                ego_lidar_pose)

            if void_lidar:
                continue

            processed_features.append(
                selected_cav_processed['processed_features'])

            infra.append(1 if selected_cav_base['params']['infra_flag'] else 0)

        # merge preprocessed features from different cavs into the same dict
        merged_feature_dict = self.merge_features_to_dict(processed_features)

        # generate the anchor boxes
        anchor_box = self.post_processor.generate_anchor_box()

        # pad dv, dt, infra to max_cav
        # velocity = velocity + (self.max_cav - len(velocity)) * [0.]
        # time_delay = time_delay + (self.max_cav - len(time_delay)) * [0.]
        infra = infra + (self.max_cav - len(infra)) * [0.]
        # spatial_correction_matrix = np.stack(spatial_correction_matrix)
        # padding_eye = np.tile(np.eye(4)[None], (self.max_cav - len(
        #     spatial_correction_matrix), 1, 1))
        # spatial_correction_matrix = np.concatenate(
        #     [spatial_correction_matrix, padding_eye], axis=0)

        processed_data_dict['ego'].update(
            {'anchor_box': anchor_box,
             'processed_lidar': merged_feature_dict,
             # 'velocity': velocity,
             # 'time_delay': time_delay,
             'infra': infra,
             # 'spatial_correction_matrix': spatial_correction_matrix
             })

        return processed_data_dict

    def get_item_single_car(self, selected_cav_base, ego_pose):
        """
        Project the lidar and bbx to ego space first, and then do clipping.

        Parameters
        ----------
        selected_cav_base : dict
            The dictionary contains a single CAV's raw information.
        ego_pose : list
            The ego vehicle lidar pose under world coordinate.

        Returns
        -------
        selected_cav_processed : dict
            The dictionary contains the cav's processed information.
        """
        selected_cav_processed = {}

        # calculate the transformation matrix,
        # no delay or loc error is considered under this setting.
        transformation_matrix = \
            x1_to_x2(selected_cav_base['params']['lidar_pose'],
                     ego_pose)

        # filter lidar
        lidar_np = selected_cav_base['lidar_np']
        lidar_np = shuffle_points(lidar_np)
        # remove points that hit itself
        lidar_np = mask_ego_points(lidar_np)
        # project the lidar to ego space
        if self.proj_first:
            lidar_np[:, :3] = \
                box_utils.project_points_by_matrix_torch(lidar_np[:, :3],
                                                         transformation_matrix)
        lidar_np = mask_points_by_range(lidar_np,
                                        self.params['preprocess'][
                                            'cav_lidar_range'])
        # Check if filtered LiDAR points are not void
        void_lidar = True if lidar_np.shape[0] < 1 else False

        processed_lidar = self.pre_processor.preprocess(lidar_np)

        # velocity
        # velocity = selected_cav_base['params']['ego_speed']
        # normalize veloccity by average speed 30 km/h
        # velocity = velocity / 30

        selected_cav_processed.update(
            {'projected_lidar': lidar_np,
             'processed_features': processed_lidar,
             })

        return selected_cav_processed, void_lidar

    @staticmethod
    def merge_features_to_dict(processed_feature_list):
        """
        Merge the preprocessed features from different cavs to the same
        dictionary.

        Parameters
        ----------
        processed_feature_list : list
            A list of dictionary containing all processed features from
            different cavs.

        Returns
        -------
        merged_feature_dict: dict
            key: feature names, value: list of features.
        """

        merged_feature_dict = OrderedDict()

        for i in range(len(processed_feature_list)):
            for feature_name, feature in processed_feature_list[i].items():
                if feature_name not in merged_feature_dict:
                    merged_feature_dict[feature_name] = []
                if isinstance(feature, list):
                    merged_feature_dict[feature_name] += feature
                else:
                    merged_feature_dict[feature_name].append(feature)

        return merged_feature_dict

    def collate_batch_train(self, batch):
        # Intermediate fusion is different the other two
        output_dict = {'ego': {}}

        processed_lidar_list = []
        # used to record different scenario
        record_len = []

        # used for PriorEncoding for models
        # velocity = []
        # time_delay = []
        infra = []


        # used for correcting the spatial transformation between delayed timestamp
        # and current timestamp
        # spatial_correction_matrix_list = []


        for i in range(len(batch)):
            ego_dict = batch[i]['ego']
            processed_lidar_list.append(ego_dict['processed_lidar'])
            # record_len.append(ego_dict['cav_num'])
            record_len.append(1)


            infra.append(ego_dict['infra'])


        # example: {'voxel_features':[np.array([1,2,3]]),
        # np.array([3,5,6]), ...]}
        merged_feature_dict = self.merge_features_to_dict(processed_lidar_list)
        processed_lidar_torch_dict = \
            self.pre_processor.collate_batch(merged_feature_dict)
        # [2, 3, 4, ..., M], M <= max_cav
        record_len = torch.from_numpy(np.array(record_len, dtype=int))

        # (B, max_cav)
        infra = torch.from_numpy(np.array(infra))
        velocity = time_delay = torch.zeros_like(infra)
        # (B, max_cav, 3)
        prior_encoding = \
            torch.stack([velocity, time_delay, infra], dim=-1).float()

        # object id is only used during inference, where batch size is 1.
        # so here we only get the first element.
        output_dict['ego'].update({'processed_lidar': processed_lidar_torch_dict,
                                   'record_len': record_len,
                                   'prior_encoding': prior_encoding})
        return output_dict

    def collate_batch_test(self, batch):
        assert len(batch) <= 1, "Batch size 1 is required during testing!"
        output_dict = self.collate_batch_train(batch)

        # check if anchor box in the batch
        if batch[0]['ego']['anchor_box'] is not None:
            output_dict['ego'].update({'anchor_box':
                torch.from_numpy(np.array(
                    batch[0]['ego'][
                        'anchor_box']))})

        # save the transformation matrix (4, 4) to ego vehicle
        transformation_matrix_torch = \
            torch.from_numpy(np.identity(4)).float()
        output_dict['ego'].update({'transformation_matrix':
                                       transformation_matrix_torch})

        return output_dict

    def post_process(self, data_dict, output_dict):
        """
        Process the outputs of the model to 2D/3D bounding box.

        Parameters
        ----------
        data_dict : dict
            The dictionary containing the origin input data of model.

        output_dict :dict
            The dictionary containing the output of the model.

        Returns
        -------
        pred_box_tensor : torch.Tensor
            The tensor of prediction bounding box after NMS.
        gt_box_tensor : torch.Tensor
            The tensor of gt bounding box.
        """
        pred_box_tensor, pred_score = \
            self.post_processor.post_process(data_dict, output_dict)

        return pred_box_tensor, pred_score, None

    def get_pairwise_transformation(self, base_data_dict, max_cav):
        """
        Get pair-wise transformation matrix accross different agents.

        Parameters
        ----------
        base_data_dict : dict
            Key : cav id, item: transformation matrix to ego, lidar points.

        max_cav : int
            The maximum number of cav, default 5

        Return
        ------
        pairwise_t_matrix : np.array
            The pairwise transformation matrix across each cav.
            shape: (L, L, 4, 4)
        """
        pairwise_t_matrix = np.zeros((max_cav, max_cav, 4, 4))

        if self.proj_first:
            # if lidar projected to ego first, then the pairwise matrix
            # becomes identity
            pairwise_t_matrix[:, :] = np.identity(4)
        else:
            warnings.warn("Projection later is not supported in "
                          "the current version. Using it will throw"
                          "an error.")
            t_list = []

            # save all transformation matrix in a list in order first.
            for cav_id, cav_content in base_data_dict.items():
                t_list.append(cav_content['params']['transformation_matrix'])

            for i in range(len(t_list)):
                for j in range(len(t_list)):
                    # identity matrix to self
                    if i == j:
                        continue
                    # i->j: TiPi=TjPj, Tj^(-1)TiPi = Pj
                    t_matrix = np.dot(np.linalg.inv(t_list[j]), t_list[i])
                    pairwise_t_matrix[i, j] = t_matrix

        return pairwise_t_matrix
