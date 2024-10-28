import gym
import numpy as np
import time

# Humanoid 환경 생성
env = gym.make("Humanoid-v4", render_mode="human")
env.reset()

# 관절 개수 확인
num_joints = env.action_space.shape[0]
print("Total joints:", num_joints)  # 관절 개수를 확인해 각 관절에 대해 제어 가능

# 특정 관절만 움직이도록 설정하는 함수
def move_joint(joint_index, strength=0.5, steps=100):
    # action 배열 초기화 (모든 관절을 움직이지 않음)
    action = np.zeros(num_joints)
    action[joint_index] = strength  # 특정 관절에만 힘을 줌

    for _ in range(steps):
        env.step(action)
        env.render()
        print(_)
        time.sleep(1)  # 속도를 조절하기 위해 잠시 대기

# 개별 관절 움직여 보기
for joint in range(num_joints):
    print(f"Moving joint {joint}")
    move_joint(joint)
    env.reset()

env.close()
