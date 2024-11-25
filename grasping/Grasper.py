import os
import logging
import numpy as np
from types import SimpleNamespace
from pydrake.all import (
    AbstractValue,
    RigidTransform,
    Context,
    LeafSystem,
    PointCloud,
    DepthImageToPointCloud,
)
import open3d as o3d
from typing import List, Tuple, Mapping

class Grasper(LeafSystem):
    def __init__(self):
        LeafSystem.__init__(self)

        # Input ports
        self._do_grasp_input = self.DeclareVectorInputPort("do_grasp", 1)
        self._grasp_pose_input = self.DeclareAbstractInputPort("grasp_pose", AbstractValue.Make(RigidTransform()))

        # Internal states
        self._done_grasp_selection = self.DeclareDiscreteState(1)
        self._is_grasping = self.DeclareDiscreteState(1)  # 0: Not grasping, 1: Grasping

        # Output ports
        self.DeclareStateOutputPort("done_grasp", self._done_grasp_selection) # TODO: Add success/fail flag
        self.DeclareStateOutputPort("is_grasping", self._is_grasping)
        # self.DeclareAbstractOutputPort("spot_desired_state", AbstractValue.Make(RigidTransform()), self.CalcDesiredArmState)

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.5, 0.0, self.Update)

    def Update(self, context, state):
        do_grasp = self._do_grasp_input.Eval(context)[0]
        is_grasping = self._is_grasping.get_vector(state).GetAtIndex(0)

        if do_grasp and not is_grasping:
            # Start grasping process
            self._is_grasping.get_mutable_vector(state).SetAtIndex(0, 1)
            self._done_grasp_selection.get_mutable_vector(state).SetAtIndex(0, 1)
            self.Grasp()

    def Grasp(self, context):
        logging.info("Grasping object")
        X_WGoal = self.GetGraspSelection(context)

        # Calc IK stuff


    def connect_ports(self, grasp_selector, builder):
        # builder.Connect( 
        #     self.get_output_port(0), # Get output port "request_grasp" from TidySpotFSM
        #     grasp_selector.GetInputPort("do_grasp")
        # )
        builder.Connect(
            grasp_selector.GetOutputPort("grasp_selection"),
            self._grasp_pose_input
        )

    def GetGraspSelection(self, grasper_context: Context) -> RigidTransform:
        return self._grasp_pose_input.Eval(grasper_context)