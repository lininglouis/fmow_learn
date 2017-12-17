# fmow_steps



确定了data prepare方式<br>


用keras跑resnet,inception<br>
用pytorch跑densenet<br>

如何将同个物体的时序数据结合在一起<br>


data_generatore

keras pytorch才能提高效率





快要生成了，数据集，
需要
1.用baseline跑模型， 检测test 提交结果，完成整个流程
2.设计自己的东西跑模型，重复
3.用pytorch并行多个模型
4.去雾 remove haze
https://flyyufelix.github.io/2016/10/11/kaggle-statefarm.html


矩阵求导 只是实质求导在形式上的一种组合
f(W) = loss.   loss 对W求导的结果，其实就是loss对w1求导得到g1, loss对w2求导得到g2, 然后把所有求导的结果组合起来G=[g1, g2, g3, ..]
然后W += G \* learning rate. 就这样而已。
也就是对矩阵求导 =  对矩阵中每个标量元素挨个求导，然后从新组成一个矩阵的形式的。
