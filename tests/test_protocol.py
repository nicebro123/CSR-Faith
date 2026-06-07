import unittest

try:
    import numpy as np
    import torch
    from tensordict import TensorDict

    from verl.protocol import DataProto
except ModuleNotFoundError:
    np = None
    torch = None
    TensorDict = None
    DataProto = None


@unittest.skipIf(DataProto is None, "protocol dependencies are not installed")
class DataProtoTest(unittest.TestCase):
    def test_reorder_keeps_tensor_and_non_tensor_batches_aligned(self):
        data = DataProto(
            batch=TensorDict({"x": torch.tensor([10, 20, 30])}, batch_size=3),
            non_tensor_batch={"label": np.array(["a", "b", "c"], dtype=object)},
        )
        data.reorder(torch.tensor([2, 0, 1]))

        self.assertEqual(data.batch["x"].tolist(), [30, 10, 20])
        self.assertEqual(data.non_tensor_batch["label"].tolist(), ["c", "a", "b"])


if __name__ == "__main__":
    unittest.main()
