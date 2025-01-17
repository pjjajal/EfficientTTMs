from multiprocessing import Value

import torch
import torch.nn.functional as F
from torch.utils.data import default_collate

_GLOBAL_SEED = 0


class RandomMaskingCollate:
    def __init__(
        self,
        img_size=(224, 224),
        memory_ps=16,
        process_ps=32,
        mask_prob=0.75,
        deterministic=True,
    ):
        self.img_size = img_size
        self.memory_ps = memory_ps
        self.process_ps = process_ps
        self.mask_prob = mask_prob

        self.m_height = self.m_width = img_size[0] // memory_ps
        self.p_height = self.p_width = img_size[0] // process_ps
        self._itr_counter = Value("i", -1)

        self.deterministic = deterministic

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        B = len(batch)

        collated_batch = default_collate(batch)

        # This is assuming the batch is a tuple of (images, labels)
        if len(collated_batch) == 2:
            images, labels = collated_batch
        else:
            images = collated_batch

        # Set the seed for the random number generator
        # This iterates the seed every time the collate function is called.
        # This ensures that the random number generator is deterministic.
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Length of tokens
        Lp = int(self.p_height * self.p_width)
        Lm = int(self.m_height * self.m_width)

        # Randomly mask the input images
        Lp_keep = int(Lp * (1 - self.mask_prob))
        Lm_keep = int(Lm * (1 - self.mask_prob))

        pnoise = torch.rand(B, Lp, generator=g if self.deterministic else None)
        mnoise = torch.rand(B, Lm, generator=g if self.deterministic else None)

        pids_shuffle = torch.argsort(pnoise, dim=1)
        mids_shuffle = torch.argsort(mnoise, dim=1)

        pids_keep = pids_shuffle[:, :Lp_keep]
        mids_keep = mids_shuffle[:, :Lm_keep]

        pids_mask = pids_shuffle[:, Lp_keep:]
        mids_mask = mids_shuffle[:, Lm_keep:]

        return collated_batch, {
            "pids_keep": pids_keep,
            "mids_keep": mids_keep,
            "pids_mask": pids_mask,
            "mids_mask": mids_mask,
        }


class RandomMatchedMaskingCollate:
    def __init__(
        self,
        img_size=(224, 224),
        memory_ps=16,
        process_ps=32,
        mask_prob=0.75,
        deterministic=True,
    ):
        self.img_size = img_size
        self.memory_ps = memory_ps
        self.process_ps = process_ps
        self.mask_prob = mask_prob

        self.m_height = self.m_width = img_size[0] // memory_ps
        self.p_height = self.p_width = img_size[0] // process_ps
        self._itr_counter = Value("i", -1)

        self.deterministic = deterministic

    def step(self):
        i = self._itr_counter
        with i.get_lock():
            i.value += 1
            v = i.value
        return v

    def __call__(self, batch):
        B = len(batch)

        collated_batch = default_collate(batch)

        # This is assuming the batch is a tuple of (images, labels)
        if len(collated_batch) == 2:
            images, labels = collated_batch
        else:
            images = collated_batch

        # Set the seed for the random number generator
        # This iterates the seed every time the collate function is called.
        # This ensures that the random number generator is deterministic.
        seed = self.step()
        g = torch.Generator()
        g.manual_seed(seed)

        # Length of tokens
        Lp = int(self.p_height * self.p_width)
        Lm = int(self.m_height * self.m_width)

        Lp_keep = int(Lp * (1 - self.mask_prob))

        pids_keep_list = []
        mids_keep_list = []
        pids_mask_list = []
        mids_mask_list = []
        for _ in range(B):
            pids_shuffle = torch.randperm(
                Lp, generator=g if self.deterministic else None
            )
            pids_keep = pids_shuffle[:Lp_keep]
            pids_mask = pids_shuffle[Lp_keep:]

            # Take the pids and create a mask.
            mask = torch.zeros(Lp)
            mask[pids_keep] = 1  # 1 is keep, 0 is mask

            mask = mask.reshape(self.p_height, self.p_width)

            # Interpolate the mask to m_height x m_width
            mmask = F.interpolate(
                mask.unsqueeze(0).unsqueeze(0),
                scale_factor=self.m_height / self.p_height,
                mode="nearest",
            ).squeeze()

            # Get the keep and mask indices of the interpolated mask.
            mmask = mmask.flatten()
            mids_keep = mmask.nonzero().flatten()
            mids_mask = (~mmask.bool()).int().nonzero().flatten()

            pids_keep_list.append(pids_keep)
            mids_keep_list.append(mids_keep)
            pids_mask_list.append(pids_mask)
            mids_mask_list.append(mids_mask)

        return collated_batch, {
            "pids_keep": default_collate(pids_keep_list),
            "mids_keep": default_collate(mids_keep_list),
            "pids_mask": default_collate(pids_mask_list),
            "mids_mask": default_collate(mids_mask_list),
        }


if __name__ == "__main__":
    import torchvision

    dataset = torchvision.datasets.FakeData(
        size=1000,
        image_size=(3, 224, 224),
        num_classes=10,
        transform=torchvision.transforms.ToTensor(),
    )

    collate_fn = RandomMatchedMaskingCollate()
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

    for batch in loader:
        (images, labels), masks = batch
        print(masks["pids_keep"].shape)
        print(masks["mids_keep"])
        break

    print(
        torch.gather(
            torch.randn(4, 49, 10),
            1,
            masks["pids_keep"].unsqueeze(-1).expand(-1, -1, 10),
        ).shape
    )
