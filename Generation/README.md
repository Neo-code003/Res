使用GAN生成动漫人物脸

作业在NTU他们自己的网址上提交

总体思路先加大epoch训练 并且每更新一次generator，更新两次discriminator，即设置n_critic=2
训练出来图片看着还不错  只是有些重复
