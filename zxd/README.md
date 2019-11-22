# 相关资料
翻译：http://www.xzhewei.com/Note-%E7%AC%94%E8%AE%B0/Video-Object-Detection/Note-Flow-Guided-Feature-Aggregation-for-Video-Object-Detection/
作者的视频解析：https://www.bilibili.com/video/av14789193/
# Install
## 安装环境
- python 2.7
- ubuntu 16
- cuda 10
- Mxnet 1.4.1
## Error
原github提到mxnet版本太低了,我们使用pip安装较新的版本`pip install mxnet-cu100==1.4.1`
### error 1
出现"a+=b"的错误，我们改成"a=a+b"
### error 2
module.py中出现`Lack a parameter in _update_params_on_kvstore()`，改成
```python
_update_params_on_kvstore(self._exec_group.param_arrays,
self._exec_group.grad_arrays,
self._kvstore,
self._param_names)
```
### error 3
`TypeError: init_params() got an unexpected keyword argument 'allow_extra'`

方法：直接去掉base_module.py中`allow_extra=allow_extra`

# 解析
训练和测试：`experiments/fgfa_rfcn/fgfa_rfcn_end2end_train_test.py` 每次运行前要删除output文件夹
测试：`experiments/fgfa_rfcn/fgfa_rfcn_test.py`  运行前在output里要有param权重文件
`experiments/fgfa_rfcn/cfgs/resnet_v1_101_flownet_imagenet_vid_rfcn_end2end_ohem.yaml`文件中的gpus控制gpu
