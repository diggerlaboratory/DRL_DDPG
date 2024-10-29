## 0. 코드를 저장소로붙터 가져오기
git clone https://github.com/diggerlaboratory/DRL_DDPG.git


## 1. 가상 환경 생성 및 활성화

```sh
conda create -n newEnv python=3.10
conda activate newEnv
```





## 2. 필요한 라이브러리 설치

* ```sh
  pip install natsort
  pip install gym==0.26
  pip install torch
  pip install 'numpy<2'
   pip install glfw
   pip install imageio
   pip install 'mujoco==2.3.3'
  ```