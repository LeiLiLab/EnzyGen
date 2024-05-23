# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging

import numpy as np
import torch
from fairseq.data import FairseqDataset, data_utils


logger = logging.getLogger(__name__)


def collate(
    samples,
    pad_idx,
    eos_idx,
    left_pad_source=False,
    left_pad_target=False,
    input_feeding=True,
    pad_to_length=None,
    pad_to_multiple=1,
):
    if len(samples) == 0:
        return {}

    def merge(key, left_pad, move_eos_to_beginning=False, pad_to_length=None):
        return data_utils.collate_tokens(
            [s[key] for s in samples],
            pad_idx,
            eos_idx,
            left_pad=False,
            move_eos_to_beginning=False,
            pad_to_length=pad_to_length,
            pad_to_multiple=pad_to_multiple,
        )

    def collate_coor(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        res = values[0].new(batch_size, size, 3).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def collate_sub_atom(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        res = values[0].new(batch_size, size, 5).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def collate_motif(
            value,
            pad_idx,
            pad_to_length=None,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        size = max(v.size(0) for v in values)
        size = size if pad_to_length is None else max(size, pad_to_length)

        batch_size = len(values)
        res = values[0].new(batch_size, size).fill_(pad_idx)

        def copy_tensor(src, dst):
            assert dst.numel() == src.numel()
            dst.copy_(src)

        for i, v in enumerate(values):
            copy_tensor(v, res[i][: len(v)])
        return res

    def collate_label(
            value,
    ):
        """Convert a list of 1d tensors into a padded 2d tensor."""
        values = [s[value] for s in samples]
        bindings = torch.LongTensor(values)
        return bindings

    id = torch.LongTensor([s["id"] for s in samples])
    src_tokens = merge(
        "source",
        left_pad=left_pad_source,
        pad_to_length=pad_to_length["source"] if pad_to_length is not None else None,
    )
    # sort by descending source length
    src_lengths = torch.LongTensor(
        [s["source"].ne(pad_idx).long().sum() for s in samples]
    )
    src_lengths, sort_order = src_lengths.sort(descending=True)
    id = id.index_select(0, sort_order)
    src_tokens = src_tokens.index_select(0, sort_order)

    target = collate_coor(
        "target",
        0,
        pad_to_length=pad_to_length["target"]
        if pad_to_length is not None
        else None,
    )
    target = target.index_select(0, sort_order)
    tgt_lengths = src_lengths
    ntokens = tgt_lengths.sum().item()

    motif_input = collate_motif(
        "motif",
        0,
        pad_to_length=pad_to_length["motif"]
        if pad_to_length is not None
        else None,
    )
    motif_output = collate_motif(
        "motif",
        0,
         pad_to_length=pad_to_length["motif"]
        if pad_to_length is not None
        else None,
    )
    motif_input = motif_input.index_select(0, sort_order)
    motif_output = motif_output.index_select(0, sort_order)

    pdb_batch = [s["pdb"] for s in samples]
    pdb_batch = [pdb_batch[index] for index in sort_order]

    ec1_batch = [s["ec1"] for s in samples]
    ec1_batch = torch.IntTensor([ec1_batch[index] for index in sort_order])

    ec2_batch = [s["ec2"] for s in samples]
    ec2_batch = torch.IntTensor([ec2_batch[index] for index in sort_order])

    ec3_batch = [s["ec3"] for s in samples]
    ec3_batch = torch.IntTensor([ec3_batch[index] for index in sort_order])

    ec4_batch = [s["ec4"] for s in samples]
    ec4_batch = torch.IntTensor([ec4_batch[index] for index in sort_order])

    center_batch = collate_motif("center", 0).index_select(0, sort_order)

    sub_coor = collate_coor("sub_coor", 0)
    sub_coor = sub_coor.index_select(0, sort_order)   # [B, L, 3]
    sub_atom = collate_sub_atom("sub_atom", 0)
    sub_atom = sub_atom.index_select(0, sort_order)   # [B, L, 5]
    sub_binding = collate_label("binding")
    sub_binding = sub_binding.index_select(0, sort_order)

    batch = {
        "id": id,
        "nsentences": len(samples),
        "ntokens": ntokens,
        "source_input": {"src_tokens": src_tokens, "src_lengths": src_lengths},
        "target_input": {"target_coor": target, "tgt_lengths": tgt_lengths},
        "substrate_input": {"substrate_coor": sub_coor, "substrate_atom": sub_atom, "binding": sub_binding},
        "motif": {"input": motif_input, "output": motif_output},
        "pdb": pdb_batch,
        "ec": {"ec1": ec1_batch, "ec2": ec2_batch, "ec3": ec3_batch, "ec4": ec4_batch},
        "center": center_batch,
    } # "ec": {"ec3": ec3_batch, "ec4": ec4_batch},
    return batch


class ProteinDataset(FairseqDataset):
    """
    A pair of torch.utils.data.Datasets.

    Args:
        src (torch.utils.data.Dataset): source dataset to wrap
        src_sizes (List[int]): source sentence lengths
        src_dict (~fairseq.data.Dictionary): source vocabulary
        tgt (torch.utils.data.Dataset, optional): target dataset to wrap
        tgt_sizes (List[int], optional): target sentence lengths
        motif:
        motif_sizes
        left_pad_source (bool, optional): pad source tensors on the left side
            (default: True).
        left_pad_target (bool, optional): pad target tensors on the left side
            (default: False).
        shuffle (bool, optional): shuffle dataset elements before batching
            (default: True).
        input_feeding (bool, optional): create a shifted version of the targets
            to be passed into the model for teacher forcing (default: True).
        remove_eos_from_source (bool, optional): if set, removes eos from end
            of source if it's present (default: False).
        append_eos_to_target (bool, optional): if set, appends eos to end of
            target if it's absent (default: False).
        align_dataset (torch.utils.data.Dataset, optional): dataset
            containing alignments.
        constraints (Tensor, optional): 2d tensor with a concatenated, zero-
            delimited list of constraints for each sentence.
        append_bos (bool, optional): if set, appends bos to the beginning of
            source/target sentence.
        num_buckets (int, optional): if set to a value greater than 0, then
            batches will be bucketed into the given number of batch shapes.
        src_lang_id (int, optional): source language ID, if set, the collated batch
            will contain a field 'src_lang_id' in 'net_input' which indicates the
            source language of the samples.
        tgt_lang_id (int, optional): target language ID, if set, the collated batch
            will contain a field 'tgt_lang_id' which indicates the target language
             of the samples.
    """

    def __init__(
        self,
        src,
        src_sizes,
        src_dict=None,
        tgt=None,
        tgt_sizes=None,
        motif=None,
        motif_sizes=None,
        pdb=None,
        ec_tree=None,
        sub_atom=None,
        sub_coor=None,
        binding=None,
        left_pad_source=False,
        left_pad_target=False,
        shuffle=True,
        input_feeding=True,
        remove_eos_from_source=False,
        append_eos_to_target=False,
        align_dataset=None,
        constraints=None,
        append_bos=False,
        eos=None,
        num_buckets=0,
        src_lang_id=None,
        tgt_lang_id=None,
        pad_to_multiple=1,
    ):
        if tgt is not None:
            assert len(src) == len(tgt) == len(motif)
        self.src = src
        self.tgt = tgt
        self.motif = motif
        self.pdb = pdb
        self.ec = ec_tree
        self.sub_atom = sub_atom
        self.sub_coor = sub_coor
        self.binding = binding
        self.src_sizes = np.array(src_sizes)
        self.tgt_sizes = np.array(tgt_sizes) if tgt_sizes is not None else None
        self.motif_sizes = np.array(motif_sizes)
        self.sizes = (
            np.vstack((self.src_sizes, self.tgt_sizes)).T
            if self.tgt_sizes is not None
            else self.src_sizes
        )
        self.src_dict = src_dict
        self.left_pad_source = left_pad_source
        self.left_pad_target = left_pad_target
        self.shuffle = shuffle
        self.input_feeding = input_feeding
        self.remove_eos_from_source = remove_eos_from_source
        self.append_eos_to_target = append_eos_to_target
        self.align_dataset = align_dataset
        if self.align_dataset is not None:
            assert (
                self.tgt_sizes is not None
            ), "Both source and target needed when alignments are provided"
        self.constraints = constraints
        self.append_bos = append_bos
        self.eos = eos if eos is not None else src_dict.eos()
        self.src_lang_id = src_lang_id
        self.tgt_lang_id = tgt_lang_id
        if num_buckets > 0:
            from fairseq.data import BucketPadLengthDataset

            self.src = BucketPadLengthDataset(
                self.src,
                sizes=self.src_sizes,
                num_buckets=num_buckets,
                pad_idx=self.src_dict.pad(),
                left_pad=self.left_pad_source,
            )
            self.src_sizes = self.src.sizes

            # determine bucket sizes using self.num_tokens, which will return
            # the padded lengths (thanks to BucketPadLengthDataset)
            num_tokens = np.vectorize(self.num_tokens, otypes=[np.compat.long])
            self.bucketed_num_tokens = num_tokens(np.arange(len(self.src)))
            self.buckets = [
                (None, num_tokens) for num_tokens in np.unique(self.bucketed_num_tokens)
            ]
        else:
            self.buckets = None
        self.pad_to_multiple = pad_to_multiple

    def get_batch_shapes(self):
        return self.buckets

    def __getitem__(self, index):
        (tgt_item, center) = self.tgt[index]
        src_item = self.src[index]
        motif_item = self.motif[index]
        pdb_item = self.pdb[index]
        (ec1_item, ec2_item, ec3_item, ec4_item) = self.ec[index]
        sub_atom = self.sub_atom[index]
        sub_coor = self.sub_coor[index]
        sub_binding = self.binding[index]

        example = {
            "id": index,
            "source": src_item,
            "target": tgt_item,
            "motif": motif_item,
            "pdb": pdb_item,
            "ec1": ec1_item,
            "ec2": ec2_item,
            "ec3": ec3_item,
            "ec4": ec4_item,
            "center": center,
            "sub_atom": sub_atom,
            "sub_coor": sub_coor,
            "binding": sub_binding
        }

        return example

    def __len__(self):
        return len(self.src)

    def collater(self, samples, pad_to_length=None):
        """Merge a list of samples to form a mini-batch.

        Args:
            samples (List[dict]): samples to collate
            pad_to_length (dict, optional): a dictionary of
                {'source': source_pad_to_length, 'target': target_pad_to_length}
                to indicate the max length to pad to in source and target respectively.

        Returns:
            dict: a mini-batch with the following keys:

                - `id` (LongTensor): example IDs in the original input order
                - `ntokens` (int): total number of tokens in the batch
                - `net_input` (dict): the input to the Model, containing keys:

                  - `src_tokens` (LongTensor): a padded 2D Tensor of tokens in
                    the source sentence of shape `(bsz, src_len)`. Padding will
                    appear on the left if *left_pad_source* is ``True``.
                  - `src_lengths` (LongTensor): 1D Tensor of the unpadded
                    lengths of each source sentence of shape `(bsz)`
                  - `prev_output_tokens` (LongTensor): a padded 2D Tensor of
                    tokens in the target sentence, shifted right by one
                    position for teacher forcing, of shape `(bsz, tgt_len)`.
                    This key will not be present if *input_feeding* is
                    ``False``.  Padding will appear on the left if
                    *left_pad_target* is ``True``.
                  - `src_lang_id` (LongTensor): a long Tensor which contains source
                    language IDs of each sample in the batch

                - `target` (LongTensor): a padded 2D Tensor of tokens in the
                  target sentence of shape `(bsz, tgt_len)`. Padding will appear
                  on the left if *left_pad_target* is ``True``.
                - `tgt_lang_id` (LongTensor): a long Tensor which contains target language
                   IDs of each sample in the batch
        """
        res = collate(
            samples,
            pad_idx=self.src_dict.pad(),
            eos_idx=self.eos,
            left_pad_source=self.left_pad_source,
            left_pad_target=self.left_pad_target,
            input_feeding=self.input_feeding,
            pad_to_length=pad_to_length,
            pad_to_multiple=self.pad_to_multiple,
        )

        return res

    def num_tokens(self, index):
        """Return the number of tokens in a sample. This value is used to
        enforce ``--max-tokens`` during batching."""
        return self.src_sizes[index]

    def num_tokens_vec(self, indices):
        """Return the number of tokens for a set of positions defined by indices.
        This value is used to enforce ``--max-tokens`` during batching."""
        sizes = self.src_sizes[indices]
        if self.tgt_sizes is not None:
            sizes = np.maximum(sizes, self.tgt_sizes[indices])
        return sizes

    def size(self, index):
        """Return an example's size as a float or tuple. This value is used when
        filtering a dataset with ``--max-positions``."""
        return (
            self.src_sizes[index],
            self.tgt_sizes[index] if self.tgt_sizes is not None else 0,
        )

    def ordered_indices(self):
        """Return an ordered list of indices. Batches will be constructed based
        on this order."""
        if self.shuffle:
            indices = np.random.permutation(len(self)).astype(np.int64)
        else:
            indices = np.arange(len(self), dtype=np.int64)
        if self.buckets is None:
            # sort by target length, then source length
            if self.tgt_sizes is not None:
                indices = indices[np.argsort(self.tgt_sizes[indices], kind="mergesort")]
            return indices[np.argsort(self.src_sizes[indices], kind="mergesort")]
        else:
            # sort by bucketed_num_tokens, which is:
            #   max(padded_src_len, padded_tgt_len)
            return indices[
                np.argsort(self.bucketed_num_tokens[indices], kind="mergesort")
            ]

    @property
    def supports_prefetch(self):
        return getattr(self.src, "supports_prefetch", False) and (
            getattr(self.tgt, "supports_prefetch", False) or self.tgt is None
        )

    def prefetch(self, indices):
        self.src.prefetch(indices)
        if self.tgt is not None:
            self.tgt.prefetch(indices)
        if self.align_dataset is not None:
            self.align_dataset.prefetch(indices)

    def filter_indices_by_size(self, indices, max_sizes):
        """Filter a list of sample indices. Remove those that are longer
            than specified in max_sizes.

        Args:
            indices (np.array): original array of sample indices
            max_sizes (int or list[int] or tuple[int]): max sample size,
                can be defined separately for src and tgt (then list or tuple)

        Returns:
            np.array: filtered sample array
            list: list of removed indices
        """
        return data_utils.filter_paired_dataset_indices_by_size(
            self.src_sizes, self.tgt_sizes, indices, max_sizes,
        )
