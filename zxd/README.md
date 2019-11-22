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
