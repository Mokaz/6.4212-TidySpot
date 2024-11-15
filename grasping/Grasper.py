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
        self.DeclareVectorInputPort("do_grasp", 1)
        self._grasp_pose_input = self.DeclareAbstractInputPort("grasp_pose", AbstractValue.Make(RigidTransform()))

        # Internal states
        self._done_grasp_selection = self.DeclareDiscreteState(1)

        # Output ports
        self.DeclareStateOutputPort("done_grasp", self._done_grasp_selection) # TODO: Add success/fail flag

        # self.DeclarePeriodicUnrestrictedUpdateEvent(0.1, 0.0, self.Update)

    def Update(self, context, state):
        do_grasp = self.EvalVectorInput(context, 0).get_value()
        if do_grasp:
            # TODO: Check if already grasping
            # Grasp() # TODO: Implement Grasp()
            self._done_grasp_selection.set_value([1])
        else:
            return

    def connect_ports(self, grasp_selector, builder):
        # builder.Connect( 
        #     self.get_output_port(0), # Get output port "request_grasp" from TidySpotPlanner
        #     grasp_selector.GetInputPort("do_grasp")
        # )
        builder.Connect(
            grasp_selector.GetOutputPort("grasp_selection"),
            self._grasp_pose_input
        )

    def GetGraspSelection(self, grasper_context: Context) -> RigidTransform:
        return self._grasp_pose_input.Eval(grasper_context)