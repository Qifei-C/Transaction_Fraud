# Detection of Transaction Fraud Risk

This project aims to detect the risk of transaction fraud, based on deep learning to classify the dataset and flag dangerous transactions.

## Summary

Transaction fraud is a type of fraudulent activity that occurs when a person or entity engages in a financial transaction with the intention of deceiving another party. This can happen in various ways, such as using stolen credit card information, creating fake accounts or identities, or manipulating the payment process in some way.

Examples of transaction fraud include:

- Unauthorized use of a credit card or debit card to make purchases or withdraw cash from an account.
- Creating a fake identity or account to open a credit card or loan account.
- Altering or manipulating invoices or receipts to inflate the amount of a transaction.
- Using phishing or other scams to trick people into giving away their financial information.

Transaction fraud can have serious consequences for both individuals and businesses. It can result in financial losses, damage to credit scores, and even legal repercussions. To prevent transaction fraud, it's important to be vigilant and protect your personal and financial information.

Through machine learning we classify payment behavior into two major categories

* Category I: Credit cards are not at risk of theft
* Category II: Credit cards are at risk of theft

## Model

Due to the low occurrence of credit card skimming instances within the dataset, training a deep learning model with the entire dataset would lead to underfitting. For instance, if credit card skimming data only represents $2\%$ of the total data, a model that simply classifies all data as non-skimming (i.e., $y\sim 1$) would achieve $98\%$ accuracy without effectively detecting credit card fraud. Consequently, it is necessary to partition the data into various training subsets based on risk levels using a clustering algorithm before training the deep learning model for classification.

### Hierarchical Clustering Model

If the user's payment behavior is visualized as an $n$-dimensional vector space $X\in\mathbb{R}^n$, the user's transaction data such as transaction time, location, amount, etc. can be represented as a point in the vector space i.e.

$$\vec{x}=(x_1,x_2,\cdots,x_n)\in\mathbf{H}\subseteq X$$

The daily transaction patterns of users, including activities such as grocery shopping, dining at local restaurants, and online shopping on popular websites, often exhibit clear repetition. This results in the projection of these consumption behaviors onto a vector space that is densely populated with numerous data points. In order to effectively process this data, **Hierarchical Clustering** is utilized, which involves measuring the similarity between data points and combining them to form a clustering tree based on their proximity. By examining the training data within the sub-tree that contains the stolen records, underfitting can be avoided by training the entire dataset directly.

Hierarchical clustering calculates the similarity in terms of the Euclidean distance in vector space. For any two points $\vec x_i$, $\vec y_i$ in vector space, their distance can be expressed as

$$D(x_i,y_i)=||x_i-y_i||_2=\sqrt{(x^i_1-y^i_1)^2+(x^i_2-y^i_2)^2+\cdots+(x^i_n-y^i_n)^2}$$

For a data set with $k$ observations, a $k\times k$ matrices is employed to calculate the distance values separately

$$\left[\begin{matrix}
0 & D(x_1,x_2) & D(x_1,x_3) & \cdots & D(x_1,x_k) \\
D(x_2,x_2) & 0 & D(x_2,x_3) & \cdots & D(x_2,x_k) \\
D(x_3,x_1) & D(x_3,x_2) & 0 & \cdots & D(x_3,x_k) \\
\vdots &\vdots & \vdots &\ddots & \vdots \\
D(x_k,x_1) & D(x_k,x_2) & D(x_k,x_3) & \cdots & 0
\end{matrix}\right]$$

In the event that two data points, namely $x_1$ and $x_2$, demonstrate a high level of similarity, where $D(x_1,x_2)\leq\text{tolerance}$, it is possible to combine the two points into a singular point. The parent point, denoted as $x^*_{(1,2)}$, will be situated at the midpoint between the two original nodes.

$$D(x_1,x^\*)=D(x^\*,x_2)=\frac{1}{2}D(x_1,x_2)$$

And thus the clustering tree is finally obtained by continuous merging.

### Deep Learning Model

After clustering, subsets often show a cluster-like distribution in Euclidean space. There are two distribution patterns for the points labeled as credit card fraud transactions in the training set, denoted by $\vec{p_i}=(p^i_1,p^i_2,\cdots,p^i_n)$:

* Category I: These points are discrete and separate from the clustered subset in the metric space.
* Category II: These points are close to the clustered subset in the metric space.

Therefore, two different deep learning algorithms will be used to separate the points labeled as credit card fraud.

For the first case, since the points labeled as credit card fraud are far away from the points representing normal transactions, during the clustering process, these points are assigned to a different subtree than the normal transaction points. We can cut the clustering tree at the $n$-th level and take the subsets $\mathbf{H}_i\subseteq\mathbf{H}$ from the $n+1$-th level, which represent the normal transactions. Then we can calculate the average coordinates $x^\*$ of these normal transactions in the vector space using Euclidean geometry, and use them as the new points in the training set. For this new dataset, a linearly separable SVM model can be used for classification.

考虑到如下形式的线性可分数据训练集：

$$\{(x_1,y_1),(x_2,y_2),\cdots,(x_n,y_n)\}:=\mathbb{H}_i(x,y)$$

其中 $x_i$ 是一个含有 $d$ 个元素的列向量，即 $x_i\in\mathbf{H}\subseteq\mathbb{R}^d$. $y_i\in\{-1,+1\}$为标量，标记数据 $x_i$ 是否为诈骗交易。$y_i=1$ 时表示 $x_i$ 属于正常交易，$y_i=-1$ 时表示 $x_i$ 为诈骗交易。我们希望分割这些点的超平面 $W$ 使得平面对于两种不同数据间的距离尽可能大，使用平行直线公式发现对于超平面 $\langle x\cdot W \rangle + b = 0$

$$margin=\rho=\frac{2}{||W||}$$

以及

$$\max\limits_{W,b}\rho\Leftrightarrow\max\limits_{W,b}\rho^2\Leftrightarrow\min\limits_{W,b}\frac{1}{2}||W||^2$$

$$\max\limits_{W,b}$$

