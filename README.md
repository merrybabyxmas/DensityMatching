<img width="490" height="254" alt="image" src="https://github.com/user-attachments/assets/3175fe3b-2ab5-438f-994f-cd5ecdc5c7f8" /># 🔧 가상환경 설정 및 실행 방법

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
<img src="https://github.com/user-attachments/assets/a85f63f1-cc59-414f-855b-b525109dea4c" width="400"/>
<img src="https://github.com/user-attachments/assets/c0e7f0ee-13cf-4e14-a484-085c2f6386e9" width="400"/>
<img src="https://github.com/user-attachments/assets/f0401ef9-966a-47f1-951b-04a7691ffa1a" width="400"/>


# 📘 Density-Matching GMM Policy Reinforcement Learning

본 프로젝트는 Humanoid-v4를 대상으로,  
정책을 Gaussian Mixture Model(GMM)로 표현하고,  
Critic이 유도하는 Q func density를 정책이 직접 모방하도록 학습하는  
Density-Matching Reinforcement Learning 알고리즘을 구현합니다.

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

<img width="490" height="254" alt="image" src="https://github.com/user-attachments/assets/a0ecb7da-1698-4f8a-ae58-fdb82af08ca0" />

<img width="3822" height="1970" alt="Image" src="https://github.com/user-attachments/assets/844a7b78-ecbf-465f-8a3a-4fc7b661a427" />
