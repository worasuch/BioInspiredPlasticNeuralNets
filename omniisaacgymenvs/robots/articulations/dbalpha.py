# Copyright (c) 2018-2022, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.


from typing import Optional

import carb
import numpy as np
import torch
from omni.isaac.core.robots.robot import Robot
from omni.isaac.core.utils.nucleus import get_assets_root_path
from omni.isaac.core.utils.stage import add_reference_to_stage
from omniisaacgymenvs.tasks.utils.usd_utils import set_drive


class Dbalpha(Robot):
    def __init__(
        self,
        prim_path: str,
        name: Optional[str] = "Dbalpha",
        usd_path: Optional[str] = None,
        translation: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:

        self._usd_path = usd_path
        self._name = name

        if self._usd_path is None:
            assets_root_path = get_assets_root_path()
            if assets_root_path is None:
                carb.log_error("Could not find Isaac Sim assets folder")
            # self._usd_path = "omniverse://localhost/Projects/dbalpha/Dbalpha.usd"
            # self._usd_path = "omniverse://localhost/Projects/dbalpha/Dbalpha_op.usd"
            # self._usd_path = "omniverse://localhost/Projects/dbalpha/Dbalpha_op_jvel1.usd"
            self._usd_path = "omniverse://localhost/Projects/dbalpha/Dbalpha_op_minjrange.usd"
        add_reference_to_stage(self._usd_path, prim_path)

        super().__init__(
            prim_path=prim_path,
            name=name,
            translation=translation,
            orientation=orientation,
            articulation_controller=None,
        )

        joint_paths = ['Thorax/BC0', 'LegThoraxLeftLink1/CF0', 'LegThoraxLeftLink2/FT0', 
                        'base_link/BC1', 'dbAlphaL2link1/CF1', 'dbAlphaL2link2/FT1',
                        'base_link/BC2', 'LegAbdomenRearLeftLink1/CF2', 'LegAbdomenRearLeftLink2/FT2',
                        'Thorax/BC3', 'LegThoraxRightLink1/CF3', 'LegThoraxRightLink2/FT3',
                        'base_link/BC4', 'LegAbdomenMidRightLink1/CF4', 'LegAbdomenMidRightLink2/FT4',
                        'base_link/BC5', 'LegAbdomenRearRightLink1/CF5', 'LegAbdomenRearRightLink2/FT5',
                       ]

        for joint_path in joint_paths:
            # print('joint_path: ', f"{self.prim_path}/{joint_path}")
            set_drive(f"{self.prim_path}/{joint_path}", "angular", "position", 0, 1, 0.1, 4.1)

