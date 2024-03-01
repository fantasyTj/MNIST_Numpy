文件结构：
final_project
==components
====__init__.py
====functions.py
====img_split.py
====layers.py
====optimizer.py
====trainer.py
==dataset
====__init__.py
====mnist.pkl
====mnist.py
==final_gui_package
====__init__.py
====final_gui.py
====final_gui.ui
====final_paintboard.py
==deepconvnet_gui.py
==deepconvnet_params.pkl
==deepconvnet.py
==train_deepconvnet

环境依赖
numpy/PIL/PyQt/pickle

使用说明
最终展示文件为 deepconvnet_gui.py ，点击运行后可根据左上角“使用指南”进行测试
最终实现的卷积深度学习网络在文件 deepconvnet.py 中，该文件调用了 components 目录下的functions.py/layers.py
对于网络的训练，专门实现了Trainer类（在components目录下的trainer文件），最后在train_deepconvnet.py中进行训练（调用deepconvnet.py/train.py/optimizer.py)
训练好的参数保存在deepconvnet.pkl中
为实现最终展示界面中多数字识别，实现了一个简单的图片分割函数，在components的img_split.py中

参考：
1.《深度学习入门》--【日】斋藤康毅
2. https://github.com/hamlinzheng/mnist
-----使用愉快-----


