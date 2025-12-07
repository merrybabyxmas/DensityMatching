# 🔧 가상환경 설정 및 실행 방법

## 1. 가상환경 생성

```bash
conda create -y -n rlenv python=3.10
conda activate rlenv

# 필요 시 (일부 환경에서 site-packages 충돌 방지)
export PYTHONNOUSERSITE=1

pip install -r requirements.txt


python run_policy.py

# Density-Matching GMM Policy Reinforcement Learning

본 프로젝트는 고차원 연속 제어 환경(Humanoid-v4)을 대상으로,  
정책을 **Gaussian Mixture Model(GMM)**로 표현하고,  
Critic이 유도하는 **Boltzmann 행동 밀도(Boltzmann Action Density)**를  
정책이 직접 모방하도록 학습하는 **Density-Matching Reinforcement Learning 알고리즘**을 구현합니다.

이를 통해 기존 Gaussian Policy가 가지는 표현력 한계를 극복하고,  
보다 안정적이고 풍부한 행동 분포를 학습할 수 있습니다.

---

# 📘 Reinforcement Learning Objective

아래는 본 프로젝트의 전체 Actor–Critic 학습 수식을 간단하게 정리한 내용입니다.

---

## 🔹 1. Action Sampling

각 상태 \( s \)에서 M개의 행동을 standard Gaussian에서 샘플링합니다:

\[
a_i \sim \mathcal{N}(0, I)
\]

---

## 🔹 2. Critic-Based Target Density

샘플 행동들에 대해 Q-score를 계산하고:

\[
Q_i = \min\big(Q_1(s,a_i), \; Q_2(s,a_i)\big)
\]

이를 softmax하여 **critic-induced Boltzmann density**를 얻습니다:

\[
p_i = \frac{\exp(Q_i / T)}{\sum_j \exp(Q_j / T)}
\]

이 \( p_i \)는 정책이 따라가야 하는 이상적인 행동 분포(target density)입니다.

---

## 🔹 3. GMM Policy Likelihood

정책은 \( K \)개의 Gaussian mixture component로 구성됩니다:

\[
\pi(a|s) = \sum_{k=1}^K w_k(s)
\,\mathcal{N}(a;\mu_k(s),\sigma_k^2(s))
\]

각 행동의 log-likelihood는 다음과 같이 계산됩니다:

\[
\log \pi(a_i|s)
\]

---

## 🔹 4. Actor Loss (Density Matching)

정책이 target density \( p_i \)를 모방하도록  
KL-divergence 기반 손실을 최소화합니다:

\[
\mathcal{L}_{KL}
= - \sum_{i=1}^{M} p_i \log \pi(a_i|s)
\]

추가적으로 entropy regularization을 적용하여  
정책이 과도하게 수축하는 것을 방지합니다:

\[
\mathcal{L}_{ent}
= -\lambda \sum_{k} w_k(s)\sum_{j} \log \sigma_{k,j}(s)
\]

최종 Actor Loss는 다음과 같습니다:

\[
\mathcal{L}_{actor}
= \mathcal{L}_{KL} + \mathcal{L}_{ent}
\]

---

## 🔹 5. Critic Loss (Soft Q-Learning)

Critic은 Soft Q-learning 방식으로 학습됩니다.

Target Q-value:

\[
y = r + \gamma (1-d)\min\big(Q_1^{tgt}(s',a'), Q_2^{tgt}(s',a')\big)
\]

Critic Loss:

\[
\mathcal{L}_{critic}
= (Q_1(s,a)-y)^2 + (Q_2(s,a)-y)^2
\]

---

## 🔹 6. Target Network Soft Update

학습 안정성을 위해 critic의 타깃 네트워크를 다음과 같이 갱신합니다:

\[
\theta^{-} \leftarrow (1-\tau)\theta^{-} + \tau\theta
\]

이 soft update는 target critic이 느리게 변화하도록 하여  
overestimation 및 학습 불안정을 방지합니다.

---

# ✨ Summary

본 알고리즘은  
- **GMM 정책의 표현력**,  
- **Density Matching의 안정적 학습 신호**,  
- **SAC 기반 critic 학습**,  
- **Target Network EMA 안정화**  

를 결합하여, Humanoid-v4와 같은 복잡한 연속 제어 환경에서  
강력하고 안정적인 성능을 제공합니다.

---

# 🧩 Repository Structure (예시)

