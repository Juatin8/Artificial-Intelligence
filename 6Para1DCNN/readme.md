## 电子烟的行为数据采集
- 这个AI项目主要是采集GPS和IMU总共六维的数据，每个维度的数据分别传入6条平行的1DCNN。打标数据是是否吸烟的0和1。
- 输出的预测结果是介于0-1的数据，可以解释为吸烟的概率。
- 服务器端用python flask架构
- ![1690434295399](https://github.com/Juatin8/Artificial-Intelligence/assets/45580031/50cefdad-52e8-491e-97fe-4c3d3fc5d700)

