# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from dataclasses import dataclass, field
import math
from omegaconf import II
import numpy as np

import torch
import torch.nn.functional as F
from fairseq import metrics, modules, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.dataclass import FairseqDataclass


@dataclass
class GeometricProteinDesignConfig(FairseqDataclass):
    tpu: bool = II("common.tpu")
    encoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the encoder loss"}
    )
    decoder_factor: float = field(
        default=1.0,
        metadata={"help": "the importance of the decoder loss"}
    )


@register_criterion("geometric_protein_loss", dataclass=GeometricProteinDesignConfig)
class GeometricProteinLoss(FairseqCriterion):
    """
    Implementation for the loss used in masked language model (MLM) training.
    """

    def __init__(self, cfg: GeometricProteinDesignConfig, task):
        super().__init__(task)
        self.tpu = cfg.tpu
        self.encoder_factor = cfg.encoder_factor
        self.decoder_factor = cfg.decoder_factor

    def forward(self, model, sample, reduce=True):
        """Compute the loss for the given sample.

        Returns a tuple with three elements:
        1) the loss
        2) the sample size, which is used as the denominator for the gradient
        3) logging outputs to display while training
        """
        source_input = sample["source_input"]
        target_input = sample["target_input"]
        motif = sample["motif"]
        output_mask = motif["output"]
        sample_size = output_mask.int().sum()
        ec = sample["ec"]

        encoder_out, decoder_out = model(source_input["src_tokens"],
                                         source_input["src_lengths"],
                                         target_input["target_coor"],
                                         motif, ec["ec1"], ec["ec2"], ec["ec3"], ec["ec4"])

        # encoder output should be logits
        loss_encoder = -torch.log(encoder_out.gather(dim=-1, index=source_input["src_tokens"].unsqueeze(-1)).squeeze(-1))
        loss_encoder = torch.mean(torch.sum(loss_encoder * output_mask, dim=-1))

        # decoder output should be the directly predicted mse loss
        target_coor = target_input["target_coor"]
        loss_decoder = torch.mean(torch.sum(torch.sum(torch.square(decoder_out - target_coor), dim=-1) * output_mask, dim=-1))
        loss = self.encoder_factor * loss_encoder + self.decoder_factor * loss_decoder
        logging_output = {
            "loss": loss.data,
            "loss_encoder": loss_encoder.data,
            "loss_decoder": loss_decoder.data,
            "ntokens": sample_size,
            "nsentences": output_mask.size()[0]}
        return loss, sample_size, logging_output

    @staticmethod
    def reduce_metrics(logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        batch_size = logging_outputs[0].get("nsentences")
        loss = sum([log["loss"].cpu() * batch_size for log in logging_outputs])
        loss_encoder = sum([log.get("loss_encoder").cpu() * batch_size for log in logging_outputs])
        loss_decoder = sum([log.get("loss_decoder").cpu() * batch_size for log in logging_outputs])
        sample_size = sum(log.get("ntokens", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss / sample_size / math.log(2), round=3
        )
        metrics.log_scalar(
            "sequence loss", loss_encoder / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "coordinate loss", loss_decoder / sample_size, sample_size, round=3
        )
        metrics.log_scalar(
            "sample size", sample_size)

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
