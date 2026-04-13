![alt text](QQ_1775652946161.png)


**Agent预处理观测并预测动作
env.step(）返回下一帧**
相当于agent预测的动作，在当前的环境下执行之后返回当前的奖励值和下一步的环境

**组装SampleData**
就是把当前这一步交互产生的所有数据打包在一起，存进collector列表。具体包括：当前观测（obs）、合法动作（legal_action）、实际执行的动作（act）、获得的奖励（reward）、是否结束（done）、价值网络的估计值（value）、动作概率（prob）。
这些数据后续在 sample_process 里会被用来计算GAE优势估计和回报，最终送给Learner做PPO更新。每一步存一个SampleData，一局下来collector里就攒了一整局的样本序列。


**sample_process() 拿到一局的原始样本序列后，做两件事：**
计算reward_sum（折扣回报）— 从最后一步往前，把未来的奖励按折扣因子γ累加，得到每一步的实际回报。这个是价值网络的训练目标。
计算advantage（GAE优势估计）— 用每一步的实际奖励、价值网络的估计值（value）和下一步的估计值（next_value），通过GAE公式算出每一步动作相对于平均水平好了多少。这个是策略网络更新的依据。
简单说就是：采样阶段只收集了原始数据，sample_process() 负责把"这个动作到底好不好"算出来，然后才能送去训练。    


send_sample_data() 是把样本传过去，真正的训练是 Algorithm.learn() 这一步——它用收到的样本做反向传播，通过clip目标函数更新策略网络，通过价值损失更新价值网络。