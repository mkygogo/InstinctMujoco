from __future__ import annotations

import argparse
import json
from collections import OrderedDict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import onnxruntime as ort


G1_JOINT_ORDER = [
    "waist_pitch_joint",
    "waist_roll_joint",
    "waist_yaw_joint",
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
]

G1_ACTION_SCALE = np.asarray(
    [
        0.43857731392336724,
        0.43857731392336724,
        0.5475464652142303,
        0.5475464652142303,
        0.3506614663788243,
        0.5475464652142303,
        0.3506614663788243,
        0.43857731392336724,
        0.43857731392336724,
        0.5475464652142303,
        0.3506614663788243,
        0.5475464652142303,
        0.3506614663788243,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.43857731392336724,
        0.07450087032950714,
        0.07450087032950714,
    ],
    dtype=np.float32,
)

BASE_ANG_VEL_SCALE = 0.25
JOINT_VEL_SCALE = 0.05
POLICY_KEYS = (
    "base_ang_vel",
    "projected_gravity",
    "velocity_commands",
    "joint_pos",
    "joint_vel",
    "actions",
    "depth_image",
)


def _build_ort_providers() -> list[str]:
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    return providers if providers else list(ort.get_available_providers())


def _run_onnx_with_batch_support(
    session: ort.InferenceSession, input_name: str, batched_input: np.ndarray
) -> np.ndarray:
    expected_batch = session.get_inputs()[0].shape[0]
    if isinstance(expected_batch, int) and expected_batch > 0 and batched_input.shape[0] != expected_batch:
        if expected_batch != 1:
            raise ValueError(
                f"ONNX model expects fixed batch={expected_batch}, but got batch={batched_input.shape[0]}."
            )
        outputs = [session.run(None, {input_name: batched_input[i : i + 1]})[0] for i in range(batched_input.shape[0])]
        return np.concatenate(outputs, axis=0)
    return session.run(None, {input_name: batched_input})[0]


def _ensure_shape(name: str, value: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch: expected {expected_shape}, got {array.shape}")
    return array


@dataclass
class PolicyFrame:
    """One control-frame worth of parkour policy inputs.

    These fields must already match InstinctMJ policy-term semantics:
    - base_ang_vel: base angular velocity in base frame, already scaled by 0.25.
    - projected_gravity: gravity direction expressed in base frame.
    - velocity_commands: the 3-D command vector used during play.
    - joint_pos: joint position relative to the reference pose in MuJoCo joint order.
    - joint_vel: joint velocity relative to the reference pose, already scaled by 0.05.
    - actions: previous raw policy action in MuJoCo joint order.
    - depth_image: preprocessed depth stack with shape (8, 18, 32).
    """

    base_ang_vel: np.ndarray
    projected_gravity: np.ndarray
    velocity_commands: np.ndarray
    joint_pos: np.ndarray
    joint_vel: np.ndarray
    actions: np.ndarray
    depth_image: np.ndarray

    def validate(self) -> None:
        _ensure_shape("base_ang_vel", self.base_ang_vel, (3,))
        _ensure_shape("projected_gravity", self.projected_gravity, (3,))
        _ensure_shape("velocity_commands", self.velocity_commands, (3,))
        _ensure_shape("joint_pos", self.joint_pos, (29,))
        _ensure_shape("joint_vel", self.joint_vel, (29,))
        _ensure_shape("actions", self.actions, (29,))
        _ensure_shape("depth_image", self.depth_image, (8, 18, 32))

    @classmethod
    def from_raw_terms(
        cls,
        *,
        base_ang_vel: Iterable[float],
        projected_gravity: Iterable[float],
        velocity_commands: Iterable[float],
        joint_pos_rel: Iterable[float],
        joint_vel_rel: Iterable[float],
        last_action: Iterable[float],
        depth_image: np.ndarray,
    ) -> "PolicyFrame":
        return cls(
            base_ang_vel=np.asarray(base_ang_vel, dtype=np.float32) * BASE_ANG_VEL_SCALE,
            projected_gravity=np.asarray(projected_gravity, dtype=np.float32),
            velocity_commands=np.asarray(velocity_commands, dtype=np.float32),
            joint_pos=np.asarray(joint_pos_rel, dtype=np.float32),
            joint_vel=np.asarray(joint_vel_rel, dtype=np.float32) * JOINT_VEL_SCALE,
            actions=np.asarray(last_action, dtype=np.float32),
            depth_image=np.asarray(depth_image, dtype=np.float32),
        )


class PolicyHistory:
    def __init__(
        self,
        history_len: int = 8,
        depth_shape: tuple[int, int, int] = (8, 18, 32),
    ):
        self.history_len = history_len
        self.depth_shape = depth_shape
        self._base_ang_vel = deque(maxlen=history_len)
        self._projected_gravity = deque(maxlen=history_len)
        self._velocity_commands = deque(maxlen=history_len)
        self._joint_pos = deque(maxlen=history_len)
        self._joint_vel = deque(maxlen=history_len)
        self._actions = deque(maxlen=history_len)
        self._depth_image = deque(maxlen=history_len)
        self.reset()

    def reset(self) -> None:
        zeros3 = np.zeros(3, dtype=np.float32)
        zeros29 = np.zeros(29, dtype=np.float32)
        zeros_depth = np.zeros(self.depth_shape, dtype=np.float32)
        self._base_ang_vel.clear()
        self._projected_gravity.clear()
        self._velocity_commands.clear()
        self._joint_pos.clear()
        self._joint_vel.clear()
        self._actions.clear()
        self._depth_image.clear()
        for _ in range(self.history_len):
            self._base_ang_vel.append(zeros3.copy())
            self._projected_gravity.append(zeros3.copy())
            self._velocity_commands.append(zeros3.copy())
            self._joint_pos.append(zeros29.copy())
            self._joint_vel.append(zeros29.copy())
            self._actions.append(zeros29.copy())
            self._depth_image.append(zeros_depth.copy())

    def append(self, frame: PolicyFrame) -> None:
        frame.validate()
        self._base_ang_vel.append(np.asarray(frame.base_ang_vel, dtype=np.float32))
        self._projected_gravity.append(np.asarray(frame.projected_gravity, dtype=np.float32))
        self._velocity_commands.append(np.asarray(frame.velocity_commands, dtype=np.float32))
        self._joint_pos.append(np.asarray(frame.joint_pos, dtype=np.float32))
        self._joint_vel.append(np.asarray(frame.joint_vel, dtype=np.float32))
        self._actions.append(np.asarray(frame.actions, dtype=np.float32))
        self._depth_image.append(np.asarray(frame.depth_image, dtype=np.float32))

    def build_policy_terms(self) -> OrderedDict[str, np.ndarray]:
        return OrderedDict(
            [
                ("base_ang_vel", np.concatenate(list(self._base_ang_vel), axis=0)),
                ("projected_gravity", np.concatenate(list(self._projected_gravity), axis=0)),
                ("velocity_commands", np.concatenate(list(self._velocity_commands), axis=0)),
                ("joint_pos", np.concatenate(list(self._joint_pos), axis=0)),
                ("joint_vel", np.concatenate(list(self._joint_vel), axis=0)),
                ("actions", np.concatenate(list(self._actions), axis=0)),
                ("depth_image", np.asarray(self._depth_image[-1], dtype=np.float32)),
            ]
        )


class PolicyNormalizer:
    def __init__(self, npz_path: Path):
        payload = np.load(npz_path, allow_pickle=True)
        self.mean = payload["mean"].astype(np.float32)
        self.std = payload["std"].astype(np.float32)
        self.eps = float(payload["eps"])

    def normalize(self, flat_obs: np.ndarray) -> np.ndarray:
        return (flat_obs - self.mean) / (self.std + self.eps)


class ParkourOnnxPolicy:
    def __init__(self, model_dir: str | Path):
        self.model_dir = Path(model_dir).expanduser().resolve()
        metadata_path = self.model_dir / "metadata.json"
        normalizer_path = self.model_dir / "policy_normalizer.npz"
        depth_encoder_path = self.model_dir / "0-depth_encoder.onnx"
        actor_path = self.model_dir / "actor.onnx"
        for required in (metadata_path, normalizer_path, depth_encoder_path, actor_path):
            if not required.exists():
                raise FileNotFoundError(f"Missing required file: {required}")

        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        self.task_id = metadata["task_id"]
        self.num_actions = int(metadata["num_actions"])
        self.policy_shapes = OrderedDict(
            (key, tuple(value)) for key, value in metadata["obs_format"]["policy"].items()
        )
        self.normalizer = PolicyNormalizer(normalizer_path)
        providers = _build_ort_providers()
        self.depth_encoder = ort.InferenceSession(str(depth_encoder_path), providers=providers)
        self.actor = ort.InferenceSession(str(actor_path), providers=providers)
        self.depth_encoder_input_name = self.depth_encoder.get_inputs()[0].name
        self.actor_input_name = self.actor.get_inputs()[0].name
        self._flat_slices = self._build_flat_slices(self.policy_shapes)
        self._depth_shape = self.policy_shapes["depth_image"]
        self._proprio_size = self._flat_slices["depth_image"].start
        self._depth_latent_size = int(np.prod(self.depth_encoder.get_outputs()[0].shape[1:]))

    @staticmethod
    def _build_flat_slices(policy_shapes: OrderedDict[str, tuple[int, ...]]) -> dict[str, slice]:
        slices: dict[str, slice] = {}
        cursor = 0
        for key, shape in policy_shapes.items():
            width = int(np.prod(shape))
            slices[key] = slice(cursor, cursor + width)
            cursor += width
        return slices

    def flatten_policy_terms(self, policy_terms: OrderedDict[str, np.ndarray]) -> np.ndarray:
        flat_parts: list[np.ndarray] = []
        for key in POLICY_KEYS:
            if key not in self.policy_shapes:
                raise KeyError(f"Missing policy key in exported metadata: {key}")
            if key not in policy_terms:
                raise KeyError(f"Missing policy term for inference: {key}")
            expected_shape = self.policy_shapes[key]
            value = _ensure_shape(key, policy_terms[key], expected_shape)
            flat_parts.append(value.reshape(-1))
        flat_obs = np.concatenate(flat_parts, axis=0, dtype=np.float32)
        return flat_obs.reshape(1, -1)

    def act(self, history: PolicyHistory) -> np.ndarray:
        policy_terms = history.build_policy_terms()
        flat_obs = self.flatten_policy_terms(policy_terms)
        normalized_obs = self.normalizer.normalize(flat_obs.astype(np.float32))

        depth_slice = self._flat_slices["depth_image"]
        depth_input = normalized_obs[:, depth_slice].reshape(-1, *self._depth_shape)
        depth_latent = _run_onnx_with_batch_support(self.depth_encoder, self.depth_encoder_input_name, depth_input)

        proprio_input = normalized_obs[:, : self._proprio_size]
        actor_input = np.concatenate([proprio_input, depth_latent], axis=1).astype(np.float32)
        actions = _run_onnx_with_batch_support(self.actor, self.actor_input_name, actor_input)
        return np.asarray(actions[0], dtype=np.float32)

    def action_to_joint_targets(
        self,
        action: np.ndarray,
        nominal_joint_pos: np.ndarray | None = None,
        action_scale: np.ndarray | None = None,
    ) -> np.ndarray:
        nominal = np.zeros(self.num_actions, dtype=np.float32) if nominal_joint_pos is None else np.asarray(
            nominal_joint_pos, dtype=np.float32
        )
        scale = G1_ACTION_SCALE if action_scale is None else np.asarray(action_scale, dtype=np.float32)
        _ensure_shape("nominal_joint_pos", nominal, (self.num_actions,))
        _ensure_shape("action_scale", scale, (self.num_actions,))
        _ensure_shape("action", np.asarray(action, dtype=np.float32), (self.num_actions,))
        return nominal + np.asarray(action, dtype=np.float32) * scale


def smoke_test(model_dir: Path) -> None:
    policy = ParkourOnnxPolicy(model_dir)
    history = PolicyHistory()
    zero_frame = PolicyFrame(
        base_ang_vel=np.zeros(3, dtype=np.float32),
        projected_gravity=np.array([0.0, 0.0, -1.0], dtype=np.float32),
        velocity_commands=np.zeros(3, dtype=np.float32),
        joint_pos=np.zeros(29, dtype=np.float32),
        joint_vel=np.zeros(29, dtype=np.float32),
        actions=np.zeros(29, dtype=np.float32),
        depth_image=np.zeros((8, 18, 32), dtype=np.float32),
    )
    history.append(zero_frame)
    action = policy.act(history)
    targets = policy.action_to_joint_targets(action)
    print(f"task_id={policy.task_id}")
    print(f"num_actions={policy.num_actions}")
    print(f"depth_latent_size={policy._depth_latent_size}")
    print("joint_order=")
    for name in G1_JOINT_ORDER:
        print(f"  {name}")
    print("action=")
    print(np.array2string(action, precision=5, suppress_small=False))
    print("joint_targets=")
    print(np.array2string(targets, precision=5, suppress_small=False))


def main() -> None:
    parser = argparse.ArgumentParser(description="Minimal InstinctMJ parkour ONNX deployment helper for MuJoCo.")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("/home/jr/ProjectInstinct/mujoco/pretrained_weights/parkour/exported"),
        help="Directory that contains metadata.json, policy_normalizer.npz, 0-depth_encoder.onnx and actor.onnx.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Run one zero-observation inference pass and print the 29-D action.",
    )
    args = parser.parse_args()

    if args.smoke_test:
        smoke_test(args.model_dir)
        return

    print("This script is a deployment helper module.")
    print("Use `--smoke-test` first, then import `ParkourOnnxPolicy` from your MuJoCo loop.")


if __name__ == "__main__":
    main()