作业2直接开始上强度

第一步简单修改网络加宽加深  加入两层drop层（老师之后课上讲的过medium baseline

concat_nframes = 1 → 19
hidden_layers = 1 → 3
hidden_dim = 256 → 1024
batch_size = 512 → 2048

第一个超参明显设定为1明显不合理  继续加大batch_size 加深加宽网络  保存的checkpoint是能过strong_baseline的
