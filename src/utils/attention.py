from vit_pytorch.recorder import Recorder, find_modules
from timm.models.vision_transformer import Attention
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision.transforms.functional import to_pil_image
import mlflow


class AttentionGetter(Recorder):
    def _register_hook(self):
        attention_module = find_modules(self.vit.blocks[-1], Attention)[0]
        module = attention_module.attn_drop
        handle = module.register_forward_hook(self._hook)
        self.hooks.append(handle)
        self.hook_registered = True


def save_attention(model, original_img, input_tensor, img_name):

    attentions = model.get_attention_maps(input_tensor, True)
    nh = attentions.shape[0]
    attentions_mean = np.mean(attentions, axis=0)

    fig = plt.figure(figsize=(10, 3), dpi=150)
    plt.subplot(1, nh+2, 1)
    plt.title("input")
    plt.imshow(original_img)
    plt.axis("off")
    for i in range(nh):
        plt.subplot(1, nh+2, i+2)
        plt.title("Head #"+str(i))
        plt.imshow(attentions[i])
        plt.axis("off")

    plt.subplot(1, nh+2, nh+2)
    plt.title("Mean")
    plt.imshow(attentions_mean)

    plt.axis("off")
    plt.tight_layout()
    mlflow.log_figure(fig, f"{img_name}.png")
    plt.close('all')


def save_attentions(model, n_clusters, cluster, dataset, n_imgs=1):
    for c in range(n_clusters):
        current = np.where(cluster == c)[0]
        choice = np.random.choice(current, n_imgs, replace=False)
        for img_i in choice:
            img_tensor, label = dataset[img_i]
            img = dataset.get_image_without_transform(img_i)
            save_attention(model, img, img_tensor, f"attention/{c:0>2}-{img_i}")
