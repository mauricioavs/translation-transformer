import torch

from masks import create_decoder_mask, create_padding_mask


def test_padding_mask_marks_non_pad_tokens():
    seq = torch.tensor([[1, 0, 2]])
    mask = create_padding_mask(seq, pad_id=0)
    expected = torch.tensor([[[[True, False, True]]]])
    assert torch.equal(mask, expected)


def test_decoder_mask_combines_padding_and_causality():
    tgt = torch.tensor([[1, 2, 0]])
    mask = create_decoder_mask(tgt, pad_id=0)
    # Only positions before or equal to the current index and not padding are True
    expected = torch.tensor(
        [
            [
                [
                    [True, False, False],
                    [True, True, False],
                    [True, True, False],
                ]
            ]
        ],
        dtype=torch.bool,
    )
    assert torch.equal(mask, expected)
