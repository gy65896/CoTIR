<div align="center">
<img align="left" width="180" height="180" src="https://github.com/gy65896/CoTIR/blob/main/img/cotir.png" alt="">
    
## [Arxiv 2026] Universal Image Restoration via Internalized<br> Chain-of-Thought Reasoning

[![ArXiv](https://img.shields.io/badge/CoTIR-ArXiv-red.svg)]()
[![Paper](https://img.shields.io/badge/CoTIR-Paper-yellow.svg)]()
[![Web](https://img.shields.io/badge/CoTIR-Web-blue.svg)](https://gy65896.github.io/projects/ArXiv2026_CoTIR/index.html)
 
</div>

<div align=center>
<img src="https://github.com/gy65896/CoTIR/blob/main/img/abstract.jpg" width="720">
</div>

---
---
>**Universal Image Restoration via Internalized Chain-of-Thought Reasoning**<br> [Yu Guo](https://scholar.google.com/citations?user=klYz-acAAAAJ&hl=zh-CN)<sup>† </sup>, [Zhengru Fang](https://scholar.google.com/citations?user=yggQMJMAAAAJ)<sup>† </sup>, [Shengfeng He](https://scholar.google.com/citations?user=rBWnK8wAAAAJ), [Senkang Hu](https://scholar.google.com/citations?user=rtPVwT8AAAAJ), [Yihang Tao](https://scholar.google.com/citations?user=YopoapwAAAAJ), [Phone Lin](https://www.csie.ntu.edu.tw/~plin/), [Yuguang Fang](https://scholar.google.com/citations?user=cs45mqMAAAAJ)<br>
(† Co-first Author)<br>
>Arxiv<br>

> **Abstract:** *Image restoration seeks to recover high-quality images from degraded inputs but becomes highly ill-posed under complex, mixed degradations. While unified all-in-one models are common, their performance declines as degradation complexity increases. Recent works adopt Chain-of-Thought (CoT) reasoning for multi-round restoration using specialized modules. However, this approach faces two key limitations: (i) increased computational cost due to multi-step processing and (ii) weak modeling of interactions between degradations during stepwise inference. We introduce CoTIR, a universal image restoration framework that internalizes CoT reasoning within a single model. Concretely, we view image restoration as a specialized subtask of image editing, which implies that a large-scale pre-trained editing model provides a more favorable optimization starting point. Building on this, we fine-tune the model for restoration and further encode structured CoT-style reasoning into the learning objective via a differentiable formulation inspired by Lagrangian optimization, enabling holistic restoration without chaining specialized restorers. To facilitate training and evaluation, we further present CoTIR-Bench, a large-scale benchmark comprising 5.2 million samples with CoT-style reasoning traces. Extensive experiments on CoTIR-Bench and broad real composite degradation scenes show that CoTIR achieves stronger perceptual quality and more competitive fidelity than both all-in-one models and multi-round restoration methods.*
---

## News 🚀
* **2026.06.16**: Code is released.


## Method Architecture

</div>
<div align=center>
<img src="https://github.com/gy65896/CoTIR/blob/main/img/method.jpg" width="1080">
</div>