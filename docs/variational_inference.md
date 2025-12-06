变分推断（Variational Inference, VI）是一种通过**函数空间优化**近似复杂概率分布的方法，核心思想是用简单的参数化分布族逼近难以解析的后验分布。

通常有以下步骤：

- **定义变分分布**：选择一个简单的分布来近似后验分布。
- **优化变分参数**：通过最大化变分下界（ELBO）来优化变分分布的参数。
- [**推断隐变量**：利用优化后的变分分布进行推断。](https://zhuanlan.zhihu.com/p/1893801387277648020)



[用变分推断统一理解生成模型（VAE、GAN、AAE、ALI） - 科学空间|Scientific Spaces](https://spaces.ac.cn/archives/5716)

$$
p(z|x) = \frac{p(x, z)}{p(x)}
\\= \frac{p(x|z)p(z)}{p(x)}
$$
后验分布： $p(z|x)$

先验分布： $p(z)$

似然分布： $p(x|z)$

边缘分布： $p(x)$

