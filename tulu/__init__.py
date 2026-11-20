"""诞临·图鲁 诞临阶段最优出牌研究。

  tulu.env     规则环境（numba），所有方法共用
  tulu.param   方法 A：参数化策略 + CEM + 滚动前瞻
  tulu.rl      方法 B：神经网络强化学习（向量化环境 + PPO）
  tulu.common  共用工具（统计、命名）
"""
__version__ = "0.3.0"
