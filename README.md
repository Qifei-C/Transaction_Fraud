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
$$
\vec{x}=(x_1,x_2,\cdots,x_n)\in\mathbf{H}\subseteq X
$$
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
In the event that two data points, namely $x_1$ and $x_2$, demonstrate a high level of similarity, where $D(x_1,x_2)<tolerance$, it is possible to combine the two points into a singular point. The parent point, denoted as $x^*_{(1,2)}$, will be situated at the midpoint between the two original nodes.
$$D(x_1,x^*)=D(x^*,x_2)=\frac{1}{2}D(x_1,x_2)$$
And thus the clustering tree is finally obtained by continuous merging.

### Deep Learning Model

