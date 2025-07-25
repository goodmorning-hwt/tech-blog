+++
title = 'Diffusion Note 0'
date = 2025-07-19T16:47:30+08:00
draft = true
+++


## 我的 Diffusion Model 学习笔记：逐行拆解代码，搞懂背后原理

大家好！最近我一头扎进了 Diffusion Model 的世界，决定从头开始手撕代码，真正搞懂这个神奇的东西是怎么工作的。最好的学习方法就是对着代码自言自语，一边看一边问，一边想一边答。

下面就是我“手撕”一个基础 Diffusion Model 训练过程的笔记。我把它整理了出来，希望能给同样在学习路上的朋友们一些启发。

### Part 1：搭建模型骨架 - UNet2DModel

一切始于模型的定义。在 `diffusers` 库中，我们可以很方便地初始化一个 U-Net 模型。它是 Diffusion Model 的核心。

```python
# Create a model
model = UNet2DModel(
    sample_size=image_size,      # the target image resolution
    in_channels=3,               # the number of input channels, 3 for RGB images
    out_channels=3,              # the number of output channels
    layers_per_block=2,          # how many ResNet layers to use per UNet block
    block_out_channels=(64, 128, 128, 256),  # More channels -> more parameters
    down_block_types=(
        "DownBlock2D",           # a regular ResNet downsampling block
        "DownBlock2D",
        "AttnDownBlock2D",       # a ResNet downsampling block with spatial self-attention
        "AttnDownBlock2D",
    ),
    up_block_types=(
        "AttnUpBlock2D",
        "AttnUpBlock2D",         # a ResNet upsampling block with spatial self-attention
        "UpBlock2D",
        "UpBlock2D",             # a regular ResNet upsampling block
    ),
)
```

看到这段代码，我问了自己好几个“为什么”。

  * **`sample_size`**: 这定义了模型处理的图片分辨率。在预处理阶段，所有图片都被调整到了 `image_size` x `image_size` 的正方形。这是大多数深度学习模型的常规操作，要求输入尺寸的统一。

  * **`in_channels=3` 和 `out_channels=3`**: 这里的逻辑很直接。我们的任务是为一张彩色的 RGB 图片去噪。模型输入的是一张带噪声的 RGB 图片（3个通道），而它的目标是预测出被添加进去的噪声，这个噪声同样作用于 RGB 三个通道，所以输出也是3个通道。

  * **U-Net 的核心思想**: U-Net 就像一个 “U” 形沙漏。在 “U” 的左侧（下采样），图片尺寸不断变小，而通道数（可以理解为模型学习到的“特征”数量）不断增加。

      * **`block_out_channels=(64, 128, 128, 256)`**: 这个元组精确地描述了通道数的变化。输入是3通道，经过第一个 block 后，通道数激增到64。在U-Net的最底部，通道数达到了最多的256。这代表模型在更抽象的层面（图片尺寸小）上提取了更丰富的特征信息。

  * **Block 的类型与 Attention 机制**:

      * **`down_block_types` 和 `up_block_types`**: 这里藏着 U-Net 实现的精髓。你会发现，`DownBlock2D`（普通下采样块）和 `AttnDownBlock2D`（带注意力机制的下采样块）被混合使用了。
      * **为什么不全是 Attention？** 因为 Attention 机制，特别是 Self-Attention，非常消耗计算资源。它的作用是捕捉图像中长距离的依赖关系，建立“全局”视野。
      * **策略性地使用 Attention**: 因此，模型的设计者做了一个权衡。在U-Net的浅层，使用普通的卷积块（`DownBlock2D`）来高效地学习局部特征。当特征图被压缩，信息更密集时（例如通道数从128到256），再引入更昂贵的 Attention 块 (`AttnDownBlock2D`) 来学习全局信息。这是一种非常聪明的、兼顾效率和效果的设计。
      * **`up_block_types`** 正好是 `down_block_types` 的镜像，它负责将这些抽象的、高维度的特征逐步“解码”，还原成图片尺寸，同时减少通道数，最终输出预测的噪声。

### Part 2：训练的核心 - Training Loop

模型搭好了，就要开始“喂”数据训练了。这一部分是整个流程的心脏。

```python
# Set the noise scheduler
noise_scheduler = DDPMScheduler(num_train_timesteps=1000, beta_schedule="squaredcos_cap_v2")

# Training loop
optimizer = torch.optim.AdamW(model.parameters(), lr=4e-4)

losses = []

for epoch in range(30):
    for step, batch in enumerate(train_dataloader):
        clean_images = batch["images"].to(device)
        # Sample noise to add to the images
        noise = torch.randn(clean_images.shape).to(clean_images.device)
        bs = clean_images.shape[0]

        # Sample a random timestep for each image
        timesteps = torch.randint(0, noise_scheduler.num_train_timesteps, (bs,), device=clean_images.device).long()

        # Add noise to the clean images according to the noise magnitude at each timestep
        noisy_images = noise_scheduler.add_noise(clean_images, noise, timesteps)

        # Get the model prediction
        noise_pred = model(noisy_images, timesteps, return_dict=False)[0]

        # Calculate the loss
        loss = F.mse_loss(noise_pred, noise)
        loss.backward() # Corrected from loss.backward(loss)
        losses.append(loss.item())

        # Update the model parameters with the optimizer
        optimizer.step()
        optimizer.zero_grad()

    if (epoch + 1) % 5 == 0:
        loss_last_epoch = sum(losses[-len(train_dataloader) :]) / len(train_dataloader)
        print(f"Epoch:{epoch+1}, loss: {loss_last_epoch}")
```

我们一步一步拆解这个循环：

1.  **`DDPMScheduler`**: 这是一个“噪声调度器”。Diffusion Model 的核心就是模拟一个从清晰到完全噪声的过程。`num_train_timesteps=1000` 定义了这个过程总共有1000步。调度器的作用就是根据当前的步数（timestep），精确地计算出应该添加多少噪声。

2.  **`optimizer`**: 我们选择了 `AdamW`，一个非常经典且强大的优化器。它的任务是根据计算出的损失（loss），来更新模型的所有可学习参数（`model.parameters()`）。`lr=4e-4` (即 $4 \\times 10^{-4}$) 是学习率，控制每次参数更新的步子大小。

3.  **循环内部**:

      * **获取数据**: `clean_images = batch["images"].to(device)` 从数据加载器中取出一批干净的图片，并用 `.to(device)` 把它们送到 GPU 上加速计算。
      * **生成噪声**: `noise = torch.randn(...)` 生成与图片尺寸完全相同的随机噪声。
      * **随机选择`timestep`**: 这是训练的关键！`timesteps = torch.randint(...)` 会从 `0` 到 `999` 之间为批次里的每一张图片随机选择一个时间步。这意味着，模型在同一次训练中，会同时学习给“微噪声”图片去噪和给“重噪声”图片去噪。这使得模型非常鲁棒。
      * **加噪**: `noisy_images = noise_scheduler.add_noise(...)` 调用调度器，根据干净图片、噪声和随机选择的 `timestep`，生成相应噪声水平的训练样本。
      * **模型预测**: `noise_pred = model(...)` 把加噪后的图片和它对应的时间步 `timestep` 一起送入 U-Net 模型。**注意：模型的目标不是预测出原始的清晰图片，而是预测出我们当初添加的 `noise`**。
      * **计算损失**: `loss = F.mse_loss(noise_pred, noise)` 使用均方误差（Mean Squared Error）来比较模型预测的噪声 `noise_pred` 和我们真实添加的噪声 `noise`。因为我们预测的是连续的像素值（噪声），这是一个回归任务，MSE 是非常适合的损失函数。
      * **反向传播与优化**:
          * `loss.backward()`: 计算损失函数关于模型所有参数的梯度。
          * `optimizer.step()`: 优化器根据刚才计算出的梯度，使用 AdamW 算法来更新模型的权重。
          * `optimizer.zero_grad()`: 这是至关重要的一步！PyTorch 的梯度是默认累加的。在下一次迭代前，我们必须清空上一轮的梯度，否则梯度会像滚雪球一样越累越大，导致训练出错。

### Part 3：准备“燃料” - 加载数据集

再好的模型也需要数据来训练。这里我们使用了经典的 `MNIST` 数据集。

```python
dataset = torchvision.datasets.MNIST(
    root="mnist/", train=True, download=True, transform=torchvision.transforms.ToTensor()
)
```

  * `torchvision.datasets.MNIST`: PyTorch 自带的计算机视觉库，可以方便地下载和使用标准数据集。
  * `root="mnist/"`: 指定数据集下载后存放的目录。
  * `train=True`: 表明我们加载的是训练集。通常，一个数据集会被分为训练集（Training Set）、验证集（Validation Set）和测试集（Test Set）。训练集用于训练模型，验证集用于调整超参数和评估模型是否过拟合，而测试集则是在模型训练完全结束后，用来评估其最终性能的“期末考试”，它在训练过程中是完全不可见的。
  * `transform=torchvision.transforms.ToTensor()`: 这是一个转换函数，它会将下载下来的 PIL 图像格式转换为 PyTorch 的 Tensor 格式，方便后续在模型中进行计算。

### 总结

通过这样一番“手撕”，我感觉对 Diffusion Model 的训练过程有了更深刻的理解。它其实就是：

1.  **一个 U-Net 模型，任务是预测噪声。**
2.  **一个噪声调度器，负责在0到T步之间精确加噪。**
3.  **一个训练循环，不断地（随机加噪 -\> 预测噪声 -\> 计算差距 -\> 更新模型），直到模型能精准地预测出任何时间步下的噪声。**

这个过程虽然听起来简单，但每个细节都蕴含着巧妙的设计。希望我的这篇笔记能帮你拨开迷雾，下次我们再一起探索更有趣的部分！

-----

### 附：个人笔记中的“勘误”与补充

你在梳理自己思路的时候，整体的理解和直觉都非常准确，这很难得！这里只是一些小的技术点，可以让描述更精确：

1.  **关于 `clean_images.shape`**: 你在思考时提到 `shape` 可能是2，但很快意识到它是一个包含多个维度的元组。说得完全正确！一个典型的图像批次张量，其 `shape` 应该是 `(batch_size, channels, height, width)`。所以 `clean_images.shape[0]` 精确地取出了批次大小 `batch_size`。

2.  **关于 `loss.backward(loss)`**: 在你的代码片段中，写的是 `loss.backward(loss)`。对于一个标量（scalar）损失值，直接调用 `loss.backward()` 就足够了。PyTorch 会自动计算所有参数相对于这个标量的梯度。`loss.backward(gradient)` 这种带参数的用法通常在更复杂的场景下使用，比如当 `loss` 是一个向量时，你需要提供一个梯度张量来计算“向量-雅可比积”。在我们的场景下，直接用 `loss.backward()` 即可。

3.  **关于数据集划分**: 你提到了训练集和测试集，并思考了验证集的作用。总结得很好！在实际项目中，将数据分为“训练-验证-测试”三部分是标准的做法，它可以有效地防止“数据泄露”（即用测试数据的信息来调整模型），从而得到对模型泛化能力更可靠的评估。

你的笔记非常棒，充满了思考的火花。继续保持！
