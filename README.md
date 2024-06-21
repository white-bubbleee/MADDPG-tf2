# MADDPG-tf2仓库

## 0.环境配置
### conda环境配置
 - `python==3.7`
 - `tensorflow-gpu==2.5.0`
 - `tensorflow_tensorflow_probability==1.14.0`
 - `gym==0.10.0`

### 多智能体强化学习仿真环境配置
在上述建立的conda环境中，到[github地址，下载到工程文件夹内](https://github.com/openai/multiagent-particle-envs)下载代码到功臣文件夹中，`cd multiagent-particle-envs`使用`pip install -e .`安装multiagent-particle-envs


## 1.MADDPG算法
运行`train_maddpg.py`

如果需要修改参数，修改`base\args_config.py`文件的`parse_args_maddpg`函数的里面定义的参数。


