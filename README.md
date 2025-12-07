# 🔧 가상환경 설정 및 실행 방법

## 1. 가상환경 생성
```bash
conda create -y -n rlenv python=3.10
conda activate rlenv

# 필요 시 (site-packages 충돌 방지)
export PYTHONNOUSERSITE=1

pip install -r requirements.txt
```

## 2. 실행
```bash
python run_policy.py
```

---

# 🧠 Density-Matching GMM Policy Reinforcement Learning

본 프로젝트는 고차원 연속 제어 환경(Humanoid-v4)을 대상으로,  
정책을 **Gaussian Mixture Model(GMM)**로 표현하고,  
Critic이 유도하는 **Boltzmann 행동 밀도**를 정책이 직접 모방하도록 학습하는  
**Density-Matching Reinforcement Learning 알고리즘**을 구현합니다.

---

# 📘 Reinforcement Learning Objective

## 🔹 1. Action Sampling

각 상태 $s$에서 M개의 행동을 샘플링합니다:

$$a_i \sim \mathcal{N}(0, I)$$

## 🔹 2. Critic-Based Target Density

$$Q_i = \min(Q_1(s,a_i), Q_2(s,a_i))$$

$$p_i = \frac{\exp(Q_i / T)}{\sum_j \exp(Q_j / T)}$$

## 🔹 3. GMM Policy Likelihood

$$\pi(a|s) = \sum_{k=1}^{K} w_k(s) \, \mathcal{N}(a;\mu_k(s),\sigma_k^2(s))$$

## 🔹 4. Actor Loss

$$\mathcal{L}_{\text{actor}} = - \sum_i p_i \log \pi(a_i|s) - \lambda \sum_k w_k(s) \sum_j \log \sigma_{k,j}(s)$$

## 🔹 5. Critic Loss

$$y = r + \gamma(1-d)\min(Q_1^{\text{tgt}}(s',a'), Q_2^{\text{tgt}}(s',a'))$$

$$\mathcal{L}_{\text{critic}} = (Q_1 - y)^2 + (Q_2 - y)^2$$

## 🔹 6. Target Network Soft Update

$$\theta^{-} \leftarrow (1-\tau)\theta^{-} + \tau\theta$$

---

<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/99c42511-7e95-44a1-a3b1-05ca91cba5ff" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/3734f728-5db1-45e3-afae-f97cee4c5ee3" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/1f9be87a-f312-464c-ba4f-3054d15b5343" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/a27f2c00-344b-4cf8-a42b-17e2a1dd0699" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/687843ef-b6dd-4848-ad2d-03992cd93725" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/58475683-7780-4417-8383-3962566d2c9c" />
<img width="5056" height="2656" alt="Image" src="https://github.com/user-attachments/assets/02ec0715-8f50-4f56-8775-5f55f4ba3feb" />


