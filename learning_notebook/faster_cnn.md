# faster_cnn

物体检测

过去的方法

select_search

找到所有的框（耗时）

将所有的框进行分类（更耗时）

现在的算法：

region proposal algorithms

定位框

```
tf.Variable(initial_value=None, trainable=True, collections=None, validate_shape=True, 
caching_device=None, name=None, variable_def=None, dtype=None, expected_shape=None, 
import_scope=None)
```