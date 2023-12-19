# copy from: https://github.com/dptech-corp/Uni-Fold
# https://github.com/dptech-corp/Uni-Fold/blob/main/unifold/modules/frame.py
"""
                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

   APPENDIX: How to apply the Apache License to your work.

      To apply the Apache License to your work, attach the following
      boilerplate notice, with the fields enclosed by brackets "[]"
      replaced with your own identifying information. (Don't include
      the brackets!)  The text should be enclosed in the appropriate
      comment syntax for the file format. We also recommend that a
      file or class name and description of purpose be included on the
      same "printed page" as the copyright notice for easier
      identification within third-party archives.

   Copyright (c) 2022, DP Technology

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Sequence, Tuple

import numpy as np
import torch


def zero_translation(
    batch_dims: Tuple[int],
    dtype: Optional[torch.dtype] = torch.float,
    device: Optional[torch.device] = torch.device("cpu"),
    requires_grad: bool = False,
) -> torch.Tensor:
    trans = torch.zeros(
        (*batch_dims, 3), dtype=dtype, device=device, requires_grad=requires_grad
    )
    return trans


# pylint: disable=bad-whitespace
_QUAT_TO_ROT = np.zeros((4, 4, 3, 3), dtype=np.float32)

_QUAT_TO_ROT[0, 0] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]  # rr
_QUAT_TO_ROT[1, 1] = [[1, 0, 0], [0, -1, 0], [0, 0, -1]]  # ii
_QUAT_TO_ROT[2, 2] = [[-1, 0, 0], [0, 1, 0], [0, 0, -1]]  # jj
_QUAT_TO_ROT[3, 3] = [[-1, 0, 0], [0, -1, 0], [0, 0, 1]]  # kk

_QUAT_TO_ROT[1, 2] = [[0, 2, 0], [2, 0, 0], [0, 0, 0]]  # ij
_QUAT_TO_ROT[1, 3] = [[0, 0, 2], [0, 0, 0], [2, 0, 0]]  # ik
_QUAT_TO_ROT[2, 3] = [[0, 0, 0], [0, 0, 2], [0, 2, 0]]  # jk

_QUAT_TO_ROT[0, 1] = [[0, 0, 0], [0, 0, -2], [0, 2, 0]]  # ir
_QUAT_TO_ROT[0, 2] = [[0, 0, 2], [0, 0, 0], [-2, 0, 0]]  # jr
_QUAT_TO_ROT[0, 3] = [[0, -2, 0], [2, 0, 0], [0, 0, 0]]  # kr

_QUAT_TO_ROT = _QUAT_TO_ROT.reshape(4, 4, 9)
_QUAT_TO_ROT_tensor = torch.from_numpy(_QUAT_TO_ROT)


_QUAT_MULTIPLY = np.zeros((4, 4, 4))
_QUAT_MULTIPLY[:, :, 0] = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, -1]]

_QUAT_MULTIPLY[:, :, 1] = [[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 0, 1], [0, 0, -1, 0]]

_QUAT_MULTIPLY[:, :, 2] = [[0, 0, 1, 0], [0, 0, 0, -1], [1, 0, 0, 0], [0, 1, 0, 0]]

_QUAT_MULTIPLY[:, :, 3] = [[0, 0, 0, 1], [0, 0, 1, 0], [0, -1, 0, 0], [1, 0, 0, 0]]

_QUAT_MULTIPLY_BY_VEC = _QUAT_MULTIPLY[:, 1:, :]
_QUAT_MULTIPLY_BY_VEC_tensor = torch.from_numpy(_QUAT_MULTIPLY_BY_VEC)


class Rotation:
    def __init__(
        self,
        mat: torch.Tensor,
    ):
        if mat.shape[-2:] != (3, 3):
            raise ValueError(f"incorrect rotation shape: {mat.shape}")
        self._mat = mat

    @staticmethod
    def identity(
        shape,
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device("cpu"),
        requires_grad: bool = False,
    ) -> Rotation:
        mat = torch.eye(3, dtype=dtype, device=device, requires_grad=requires_grad)
        mat = mat.view(*((1,) * len(shape)), 3, 3)
        mat = mat.expand(*shape, -1, -1)
        return Rotation(mat)

    @staticmethod
    def mat_mul_mat(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return (a.float() @ b.float()).type(a.dtype)

    @staticmethod
    def mat_mul_vec(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return (r.float() @ t.float().unsqueeze(-1)).squeeze(-1).type(t.dtype)

    def __getitem__(self, index: Any) -> Rotation:
        if not isinstance(index, tuple):
            index = (index,)
        return Rotation(mat=self._mat[index + (slice(None), slice(None))])

    def __mul__(self, right: Any) -> Rotation:
        if isinstance(right, (int, float)):
            return Rotation(mat=self._mat * right)
        elif isinstance(right, torch.Tensor):
            return Rotation(mat=self._mat * right[..., None, None])
        else:
            raise TypeError(
                f"multiplicand must be a tensor or a number, got {type(right)}."
            )

    def __rmul__(self, left: Any) -> Rotation:
        return self.__mul__(left)

    def __matmul__(self, other: Rotation) -> Rotation:
        new_mat = Rotation.mat_mul_mat(self.rot_mat, other.rot_mat)
        return Rotation(mat=new_mat)

    @property
    def _inv_mat(self):
        return self._mat.transpose(-1, -2)

    @property
    def rot_mat(self) -> torch.Tensor:
        return self._mat

    def invert(self) -> Rotation:
        return Rotation(mat=self._inv_mat)

    def apply(self, pts: torch.Tensor) -> torch.Tensor:
        return Rotation.mat_mul_vec(self._mat, pts)

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        return Rotation.mat_mul_vec(self._inv_mat, pts)

    # inherit tensor behaviors
    @property
    def shape(self) -> torch.Size:
        s = self._mat.shape[:-2]
        return s

    @property
    def dtype(self) -> torch.dtype:
        return self._mat.dtype

    @property
    def device(self) -> torch.device:
        return self._mat.device

    @property
    def requires_grad(self) -> bool:
        return self._mat.requires_grad

    def unsqueeze(self, dim: int) -> Rotation:
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")

        rot_mats = self._mat.unsqueeze(dim if dim >= 0 else dim - 2)
        return Rotation(mat=rot_mats)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Rotation:
        mat = self._mat.view(self._mat.shape[:-2] + (9,))
        mat = torch.stack(list(map(fn, torch.unbind(mat, dim=-1))), dim=-1)
        mat = mat.view(mat.shape[:-1] + (3, 3))
        return Rotation(mat=mat)

    @staticmethod
    def cat(rs: Sequence[Rotation], dim: int) -> Rotation:
        rot_mats = [r.rot_mat for r in rs]
        rot_mats = torch.cat(rot_mats, dim=dim if dim >= 0 else dim - 2)

        return Rotation(mat=rot_mats)

    def cuda(self) -> Rotation:
        return Rotation(mat=self._mat.cuda())

    def to(
        self, device: Optional[torch.device], dtype: Optional[torch.dtype]
    ) -> Rotation:
        return Rotation(mat=self._mat.to(device=device, dtype=dtype))

    def type(self, dtype: Optional[torch.dtype]) -> Rotation:
        return Rotation(mat=self._mat.type(dtype))

    def detach(self) -> Rotation:
        return Rotation(mat=self._mat.detach())


class Frame:
    def __init__(
        self,
        rotation: Optional[Rotation],
        translation: Optional[torch.Tensor],
    ):
        if rotation is None and translation is None:
            rotation = Rotation.identity((0,))
            translation = zero_translation((0,))
        elif translation is None:
            translation = zero_translation(
                rotation.shape, rotation.dtype, rotation.device, rotation.requires_grad
            )

        elif rotation is None:
            rotation = Rotation.identity(
                translation.shape[:-1],
                translation.dtype,
                translation.device,
                translation.requires_grad,
            )

        if (rotation.shape != translation.shape[:-1]) or (
            rotation.device != translation.device
        ):
            raise ValueError("RotationMatrix and translation incompatible")

        self._r = rotation
        self._t = translation

    @staticmethod
    def identity(
        shape: Iterable[int],
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device("cpu"),
        requires_grad: bool = False,
    ) -> Frame:
        return Frame(
            Rotation.identity(shape, dtype, device, requires_grad),
            zero_translation(shape, dtype, device, requires_grad),
        )

    def __getitem__(
        self,
        index: Any,
    ) -> Frame:
        if type(index) != tuple:
            index = (index,)

        return Frame(
            self._r[index],
            self._t[index + (slice(None),)],
        )

    def __mul__(
        self,
        right: torch.Tensor,
    ) -> Frame:
        if not (isinstance(right, torch.Tensor)):
            raise TypeError("The other multiplicand must be a Tensor")

        new_rots = self._r * right
        new_trans = self._t * right[..., None]

        return Frame(new_rots, new_trans)

    def __rmul__(
        self,
        left: torch.Tensor,
    ) -> Frame:
        return self.__mul__(left)

    @property
    def shape(self) -> torch.Size:
        s = self._t.shape[:-1]
        return s

    @property
    def device(self) -> torch.device:
        return self._t.device

    def get_rots(self) -> Rotation:
        return self._r

    def get_trans(self) -> torch.Tensor:
        return self._t

    def compose(
        self,
        other: Frame,
    ) -> Frame:
        new_rot = self._r @ other._r
        new_trans = self._r.apply(other._t) + self._t
        return Frame(new_rot, new_trans)

    def apply(
        self,
        pts: torch.Tensor,
    ) -> torch.Tensor:
        rotated = self._r.apply(pts)
        return rotated + self._t

    def invert_apply(self, pts: torch.Tensor) -> torch.Tensor:
        pts = pts - self._t
        return self._r.invert_apply(pts)

    def invert(self) -> Frame:
        rot_inv = self._r.invert()
        trn_inv = rot_inv.apply(self._t)

        return Frame(rot_inv, -1 * trn_inv)

    def map_tensor_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Frame:
        new_rots = self._r.map_tensor_fn(fn)
        new_trans = torch.stack(list(map(fn, torch.unbind(self._t, dim=-1))), dim=-1)

        return Frame(new_rots, new_trans)

    def to_tensor_4x4(self) -> torch.Tensor:
        tensor = self._t.new_zeros((*self.shape, 4, 4))
        tensor[..., :3, :3] = self._r.rot_mat
        tensor[..., :3, 3] = self._t
        tensor[..., 3, 3] = 1
        return tensor

    @staticmethod
    def from_tensor_4x4(t: torch.Tensor) -> Frame:
        if t.shape[-2:] != (4, 4):
            raise ValueError("Incorrectly shaped input tensor")

        rots = Rotation(mat=t[..., :3, :3])
        trans = t[..., :3, 3]

        return Frame(rots, trans)

    @staticmethod
    def from_3_points(
        p_neg_x_axis: torch.Tensor,
        origin: torch.Tensor,
        p_xy_plane: torch.Tensor,
        eps: float = 1e-8,
    ) -> Frame:
        p_neg_x_axis = torch.unbind(p_neg_x_axis, dim=-1)
        origin = torch.unbind(origin, dim=-1)
        p_xy_plane = torch.unbind(p_xy_plane, dim=-1)

        e0 = [c1 - c2 for c1, c2 in zip(origin, p_neg_x_axis)]
        e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane, origin)]

        denom = torch.sqrt(sum((c * c for c in e0)) + eps)
        e0 = [c / denom for c in e0]
        dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
        e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
        denom = torch.sqrt(sum((c * c for c in e1)) + eps)
        e1 = [c / denom for c in e1]
        e2 = [
            e0[1] * e1[2] - e0[2] * e1[1],
            e0[2] * e1[0] - e0[0] * e1[2],
            e0[0] * e1[1] - e0[1] * e1[0],
        ]

        rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
        rots = rots.reshape(rots.shape[:-1] + (3, 3))

        rot_obj = Rotation(mat=rots)

        return Frame(rot_obj, torch.stack(origin, dim=-1))

    def unsqueeze(
        self,
        dim: int,
    ) -> Frame:
        if dim >= len(self.shape):
            raise ValueError("Invalid dimension")
        rots = self._r.unsqueeze(dim)
        trans = self._t.unsqueeze(dim if dim >= 0 else dim - 1)

        return Frame(rots, trans)

    @staticmethod
    def cat(
        Ts: Sequence[Frame],
        dim: int,
    ) -> Frame:
        rots = Rotation.cat([T._r for T in Ts], dim)
        trans = torch.cat([T._t for T in Ts], dim=dim if dim >= 0 else dim - 1)

        return Frame(rots, trans)

    def apply_rot_fn(self, fn: Callable[[Rotation], Rotation]) -> Frame:
        return Frame(fn(self._r), self._t)

    def apply_trans_fn(self, fn: Callable[[torch.Tensor], torch.Tensor]) -> Frame:
        return Frame(self._r, fn(self._t))

    def scale_translation(self, trans_scale_factor: float) -> Frame:
        fn = lambda t: t * trans_scale_factor
        return self.apply_trans_fn(fn)

    def stop_rot_gradient(self) -> Frame:
        fn = lambda r: r.detach()
        return self.apply_rot_fn(fn)

    @staticmethod
    def make_transform_from_reference(n_xyz, ca_xyz, c_xyz, eps=1e-20):
        input_dtype = ca_xyz.dtype
        n_xyz = n_xyz.float()
        ca_xyz = ca_xyz.float()
        c_xyz = c_xyz.float()
        n_xyz = n_xyz - ca_xyz
        c_xyz = c_xyz - ca_xyz

        c_x, c_y, d_pair = [c_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + c_x**2 + c_y**2)
        sin_c1 = -c_y / norm
        cos_c1 = c_x / norm

        c1_rots = sin_c1.new_zeros((*sin_c1.shape, 3, 3))
        c1_rots[..., 0, 0] = cos_c1
        c1_rots[..., 0, 1] = -1 * sin_c1
        c1_rots[..., 1, 0] = sin_c1
        c1_rots[..., 1, 1] = cos_c1
        c1_rots[..., 2, 2] = 1

        norm = torch.sqrt(eps + c_x**2 + c_y**2 + d_pair**2)
        sin_c2 = d_pair / norm
        cos_c2 = torch.sqrt(c_x**2 + c_y**2) / norm

        c2_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        c2_rots[..., 0, 0] = cos_c2
        c2_rots[..., 0, 2] = sin_c2
        c2_rots[..., 1, 1] = 1
        c2_rots[..., 2, 0] = -1 * sin_c2
        c2_rots[..., 2, 2] = cos_c2

        c_rots = Rotation.mat_mul_mat(c2_rots, c1_rots)
        n_xyz = Rotation.mat_mul_vec(c_rots, n_xyz)

        _, n_y, n_z = [n_xyz[..., i] for i in range(3)]
        norm = torch.sqrt(eps + n_y**2 + n_z**2)
        sin_n = -n_z / norm
        cos_n = n_y / norm

        n_rots = sin_c2.new_zeros((*sin_c2.shape, 3, 3))
        n_rots[..., 0, 0] = 1
        n_rots[..., 1, 1] = cos_n
        n_rots[..., 1, 2] = -1 * sin_n
        n_rots[..., 2, 1] = sin_n
        n_rots[..., 2, 2] = cos_n

        rots = Rotation.mat_mul_mat(n_rots, c_rots)

        rots = rots.transpose(-1, -2)
        rot_obj = Rotation(mat=rots.type(input_dtype))

        return Frame(rot_obj, ca_xyz.type(input_dtype))

    def cuda(self) -> Frame:
        return Frame(self._r.cuda(), self._t.cuda())

    @property
    def dtype(self) -> torch.dtype:
        assert self._r.dtype == self._t.dtype
        return self._r.dtype

    def type(self, dtype) -> Frame:
        return Frame(self._r.type(dtype), self._t.type(dtype))


class Quaternion:
    def __init__(self, quaternion: torch.Tensor, translation: torch.Tensor):
        if quaternion.shape[-1] != 4:
            raise ValueError(f"incorrect quaternion shape: {quaternion.shape}")
        self._q = quaternion
        self._t = translation

    @staticmethod
    def identity(
        shape: Iterable[int],
        dtype: Optional[torch.dtype] = torch.float,
        device: Optional[torch.device] = torch.device("cpu"),
        requires_grad: bool = False,
    ) -> Quaternion:
        trans = zero_translation(shape, dtype, device, requires_grad)
        quats = torch.zeros(
            (*shape, 4), dtype=dtype, device=device, requires_grad=requires_grad
        )
        with torch.no_grad():
            quats[..., 0] = 1
        return Quaternion(quats, trans)

    def get_quats(self):
        return self._q

    def get_trans(self):
        return self._t

    def get_rot_mats(self):
        quats = self.get_quats()
        rot_mats = Quaternion.quat_to_rot(quats)
        return rot_mats

    @staticmethod
    def quat_to_rot(normalized_quat):
        global _QUAT_TO_ROT_tensor
        dtype = normalized_quat.dtype
        normalized_quat = normalized_quat.float()
        if _QUAT_TO_ROT_tensor.device != normalized_quat.device:
            _QUAT_TO_ROT_tensor = _QUAT_TO_ROT_tensor.to(normalized_quat.device)
        rot_tensor = torch.sum(
            _QUAT_TO_ROT_tensor
            * normalized_quat[..., :, None, None]
            * normalized_quat[..., None, :, None],
            dim=(-3, -2),
        )
        rot_tensor = rot_tensor.type(dtype)
        rot_tensor = rot_tensor.view(*rot_tensor.shape[:-1], 3, 3)
        return rot_tensor

    @staticmethod
    def normalize_quat(quats):
        dtype = quats.dtype
        quats = quats.float()
        quats = quats / torch.linalg.norm(quats, dim=-1, keepdim=True)
        quats = quats.type(dtype)
        return quats

    @staticmethod
    def quat_multiply_by_vec(quat, vec):
        dtype = quat.dtype
        quat = quat.float()
        vec = vec.float()
        global _QUAT_MULTIPLY_BY_VEC_tensor
        if _QUAT_MULTIPLY_BY_VEC_tensor.device != quat.device:
            _QUAT_MULTIPLY_BY_VEC_tensor = _QUAT_MULTIPLY_BY_VEC_tensor.to(quat.device)
        mat = _QUAT_MULTIPLY_BY_VEC_tensor
        reshaped_mat = mat.view((1,) * len(quat.shape[:-1]) + mat.shape)
        return torch.sum(
            reshaped_mat * quat[..., :, None, None] * vec[..., None, :, None],
            dim=(-3, -2),
        ).type(dtype)

    def compose_q_update_vec(
        self, q_update_vec: torch.Tensor, normalize_quats: bool = True
    ) -> torch.Tensor:
        quats = self.get_quats()
        new_quats = quats + Quaternion.quat_multiply_by_vec(quats, q_update_vec)
        if normalize_quats:
            new_quats = Quaternion.normalize_quat(new_quats)
        return new_quats

    def compose_update_vec(
        self,
        update_vec: torch.Tensor,
        pre_rot_mat: Rotation,
    ) -> Quaternion:
        q_vec, t_vec = update_vec[..., :3], update_vec[..., 3:]
        new_quats = self.compose_q_update_vec(q_vec)

        trans_update = pre_rot_mat.apply(t_vec)
        new_trans = self._t + trans_update

        return Quaternion(new_quats, new_trans)

    def stop_rot_gradient(self) -> Quaternion:
        return Quaternion(self._q.detach(), self._t)


# from: https://github.com/dptech-corp/Uni-Fold/blob/main/unifold/data/residue_constants.py
rigid_group_atom_positions = {
    "A": [
        ["N", 0, (-0.525, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.529, -0.774, -1.205)],
        ["O", 3, (0.627, 1.062, 0.000)],
    ],
    "R": [
        ["N", 0, (-0.524, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.524, -0.778, -1.209)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.616, 1.390, -0.000)],
        ["CD", 5, (0.564, 1.414, 0.000)],
        ["NE", 6, (0.539, 1.357, -0.000)],
        ["NH1", 7, (0.206, 2.301, 0.000)],
        ["NH2", 7, (2.078, 0.978, -0.000)],
        ["CZ", 7, (0.758, 1.093, -0.000)],
    ],
    "N": [
        ["N", 0, (-0.536, 1.357, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.531, -0.787, -1.200)],
        ["O", 3, (0.625, 1.062, 0.000)],
        ["CG", 4, (0.584, 1.399, 0.000)],
        ["ND2", 5, (0.593, -1.188, 0.001)],
        ["OD1", 5, (0.633, 1.059, 0.000)],
    ],
    "D": [
        ["N", 0, (-0.525, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, 0.000, -0.000)],
        ["CB", 0, (-0.526, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.593, 1.398, -0.000)],
        ["OD1", 5, (0.610, 1.091, 0.000)],
        ["OD2", 5, (0.592, -1.101, -0.003)],
    ],
    "C": [
        ["N", 0, (-0.522, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, 0.000)],
        ["CB", 0, (-0.519, -0.773, -1.212)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["SG", 4, (0.728, 1.653, 0.000)],
    ],
    "Q": [
        ["N", 0, (-0.526, 1.361, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.779, -1.207)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.615, 1.393, 0.000)],
        ["CD", 5, (0.587, 1.399, -0.000)],
        ["NE2", 6, (0.593, -1.189, -0.001)],
        ["OE1", 6, (0.634, 1.060, 0.000)],
    ],
    "E": [
        ["N", 0, (-0.528, 1.361, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, -0.000, -0.000)],
        ["CB", 0, (-0.526, -0.781, -1.207)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG", 4, (0.615, 1.392, 0.000)],
        ["CD", 5, (0.600, 1.397, 0.000)],
        ["OE1", 6, (0.607, 1.095, -0.000)],
        ["OE2", 6, (0.589, -1.104, -0.001)],
    ],
    "G": [
        ["N", 0, (-0.572, 1.337, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.517, -0.000, -0.000)],
        [],
        ["O", 3, (0.626, 1.062, -0.000)],
    ],
    "H": [
        ["N", 0, (-0.527, 1.360, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.525, -0.778, -1.208)],
        ["O", 3, (0.625, 1.063, 0.000)],
        ["CG", 4, (0.600, 1.370, -0.000)],
        ["CD2", 5, (0.889, -1.021, 0.003)],
        ["ND1", 5, (0.744, 1.160, -0.000)],
        ["CE1", 5, (2.030, 0.851, 0.002)],
        ["NE2", 5, (2.145, -0.466, 0.004)],
    ],
    "I": [
        ["N", 0, (-0.493, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.536, -0.793, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.534, 1.437, -0.000)],
        ["CG2", 4, (0.540, -0.785, -1.199)],
        ["CD1", 5, (0.619, 1.391, 0.000)],
    ],
    "L": [
        ["N", 0, (-0.520, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.773, -1.214)],
        ["O", 3, (0.625, 1.063, -0.000)],
        ["CG", 4, (0.678, 1.371, 0.000)],
        ["CD1", 5, (0.530, 1.430, -0.000)],
        ["CD2", 5, (0.535, -0.774, 1.200)],
    ],
    "K": [
        ["N", 0, (-0.526, 1.362, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, 0.000)],
        ["CB", 0, (-0.524, -0.778, -1.208)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.619, 1.390, 0.000)],
        ["CD", 5, (0.559, 1.417, 0.000)],
        ["CE", 6, (0.560, 1.416, 0.000)],
        ["NZ", 7, (0.554, 1.387, 0.000)],
    ],
    "M": [
        ["N", 0, (-0.521, 1.364, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, 0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.210)],
        ["O", 3, (0.625, 1.062, -0.000)],
        ["CG", 4, (0.613, 1.391, -0.000)],
        ["SD", 5, (0.703, 1.695, 0.000)],
        ["CE", 6, (0.320, 1.786, -0.000)],
    ],
    "F": [
        ["N", 0, (-0.518, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, 0.000, -0.000)],
        ["CB", 0, (-0.525, -0.776, -1.212)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.377, 0.000)],
        ["CD1", 5, (0.709, 1.195, -0.000)],
        ["CD2", 5, (0.706, -1.196, 0.000)],
        ["CE1", 5, (2.102, 1.198, -0.000)],
        ["CE2", 5, (2.098, -1.201, -0.000)],
        ["CZ", 5, (2.794, -0.003, -0.001)],
    ],
    "P": [
        ["N", 0, (-0.566, 1.351, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, 0.000)],
        ["CB", 0, (-0.546, -0.611, -1.293)],
        ["O", 3, (0.621, 1.066, 0.000)],
        ["CG", 4, (0.382, 1.445, 0.0)],
        # ['CD', 5, (0.427, 1.440, 0.0)],
        ["CD", 5, (0.477, 1.424, 0.0)],  # manually made angle 2 degrees larger
    ],
    "S": [
        ["N", 0, (-0.529, 1.360, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, -0.000)],
        ["CB", 0, (-0.518, -0.777, -1.211)],
        ["O", 3, (0.626, 1.062, -0.000)],
        ["OG", 4, (0.503, 1.325, 0.000)],
    ],
    "T": [
        ["N", 0, (-0.517, 1.364, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.526, 0.000, -0.000)],
        ["CB", 0, (-0.516, -0.793, -1.215)],
        ["O", 3, (0.626, 1.062, 0.000)],
        ["CG2", 4, (0.550, -0.718, -1.228)],
        ["OG1", 4, (0.472, 1.353, 0.000)],
    ],
    "W": [
        ["N", 0, (-0.521, 1.363, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.525, -0.000, 0.000)],
        ["CB", 0, (-0.523, -0.776, -1.212)],
        ["O", 3, (0.627, 1.062, 0.000)],
        ["CG", 4, (0.609, 1.370, -0.000)],
        ["CD1", 5, (0.824, 1.091, 0.000)],
        ["CD2", 5, (0.854, -1.148, -0.005)],
        ["CE2", 5, (2.186, -0.678, -0.007)],
        ["CE3", 5, (0.622, -2.530, -0.007)],
        ["NE1", 5, (2.140, 0.690, -0.004)],
        ["CH2", 5, (3.028, -2.890, -0.013)],
        ["CZ2", 5, (3.283, -1.543, -0.011)],
        ["CZ3", 5, (1.715, -3.389, -0.011)],
    ],
    "Y": [
        ["N", 0, (-0.522, 1.362, 0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.524, -0.000, -0.000)],
        ["CB", 0, (-0.522, -0.776, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG", 4, (0.607, 1.382, -0.000)],
        ["CD1", 5, (0.716, 1.195, -0.000)],
        ["CD2", 5, (0.713, -1.194, -0.001)],
        ["CE1", 5, (2.107, 1.200, -0.002)],
        ["CE2", 5, (2.104, -1.201, -0.003)],
        ["OH", 5, (4.168, -0.002, -0.005)],
        ["CZ", 5, (2.791, -0.001, -0.003)],
    ],
    "V": [
        ["N", 0, (-0.494, 1.373, -0.000)],
        ["CA", 0, (0.000, 0.000, 0.000)],
        ["C", 0, (1.527, -0.000, -0.000)],
        ["CB", 0, (-0.533, -0.795, -1.213)],
        ["O", 3, (0.627, 1.062, -0.000)],
        ["CG1", 4, (0.540, 1.429, -0.000)],
        ["CG2", 4, (0.533, -0.776, 1.203)],
    ],
}

# same order as featurizer A2I
ALPHABET = "ACDEFGHIKLMNPQRSTVWY"
A2I = {a: i for i, a in enumerate(ALPHABET)}
aa_bb_positions = [
    torch.tensor(
        [
            rigid_group_atom_positions[aa][0][-1],  # N
            rigid_group_atom_positions[aa][1][-1],  # CA
            rigid_group_atom_positions[aa][2][-1],  # C
            rigid_group_atom_positions[aa][4][-1],  # O
        ]
    ).float()
    for aa in ALPHABET
]
aa_bb_positions = torch.stack(aa_bb_positions, dim=0)
aa_bb_positions_mean = torch.ones_like(aa_bb_positions) * aa_bb_positions.mean(
    dim=0, keepdim=True
)
